# Databricks notebook source
# MAGIC %md
# MAGIC # 02_silver — typed, deduplicated game- and move-grain tables
# MAGIC
# MAGIC Reads every snapshot row in `bronze.games_raw`, keeps the **latest
# MAGIC snapshot per (game_uuid, account)**, types all fields, parses the PGN
# MAGIC movetext into per-ply rows, and MERGEs into:
# MAGIC
# MAGIC - `silver.games` — one row per game per account-perspective (ALL games:
# MAGIC   variants, unrated, and friend games included, with flags)
# MAGIC - `silver.moves` — one row per half-move with clock time (standard
# MAGIC   chess only; variant movetext like bughouse piece-drops is a
# MAGIC   different grammar and nothing downstream consumes it)
# MAGIC
# MAGIC Idempotent by construction: bronze is append-only truth, this notebook
# MAGIC recomputes from it in full (trivial at this scale) and MERGE makes the
# MAGIC write an upsert. If parsing logic changes, drop the silver tables and
# MAGIC re-run — never patch data in place.

# COMMAND ----------

from pyspark.sql import functions as F, Window
from delta.tables import DeltaTable

CATALOG = "chess"

# Both of my accounts (lowercase). Used to derive color/result perspective
# and to flag self-play (account vs. account) games.
ACCOUNTS = ["cosmos_iv", "cosmossolitarus"]

# Friendly-match opponents (lowercase). A flag, not a filter: friend games
# stay in silver so the dashboard can compare them against ranked play.
FRIENDS = [
    "as7rixx", "treyert12358", "gravityrebel",
    "papadabear514", "zepthro", "flippjc", "ripjawe",
]

# Chess.com result codes that mean the game was drawn. "win" means won;
# everything else (checkmated, timeout, resigned, abandoned, ...) is a loss.
DRAW_CODES = [
    "agreed", "repetition", "stalemate",
    "insufficient", "50move", "timevsinsufficient",
]

# COMMAND ----------

# ---- Explode snapshots to game rows, keep latest snapshot per key --------
# The same game appears in multiple snapshot files (trailing-month re-pulls),
# so rank snapshots per (game_uuid, account) and keep the freshest.

bronze = spark.table(f"{CATALOG}.bronze.games_raw")

exploded = (
    bronze
    .withColumn("account", F.regexp_extract("_source_file", r"username=([^/]+)", 1))
    .select("account", "_source_file", "_file_modified_at", "_ingested_at",
            F.explode("games").alias("g"))
    .withColumn("game_uuid", F.coalesce(F.col("g.uuid"), F.col("g.url")))
)

latest = (
    exploded
    .withColumn(
        "_rn",
        F.row_number().over(
            Window.partitionBy("game_uuid", "account")
                  .orderBy(F.desc("_file_modified_at"), F.desc("_ingested_at"))
        ),
    )
    .filter("_rn = 1")
    .drop("_rn")
)

# COMMAND ----------

# ---- silver.games: one typed row per game per account-perspective --------

is_white = F.lower(F.col("g.white.username")) == F.col("account")
my = F.when(is_white, F.col("g.white")).otherwise(F.col("g.black"))
opp = F.when(is_white, F.col("g.black")).otherwise(F.col("g.white"))

tc = F.col("g.time_control")

games_df = (
    latest.select(
        "game_uuid",
        "account",
        F.col("g.url").alias("url"),
        F.col("g.rated").cast("boolean").alias("rated"),
        F.col("g.time_class").alias("time_class"),
        F.col("g.rules").alias("rules"),
        tc.alias("time_control"),
        # "600" -> 600/0 ; "600+5" -> 600/5 ; daily "1/259200" -> null/null
        F.when(tc.rlike(r"^[0-9]+$"), tc.cast("int"))
         .when(tc.rlike(r"^[0-9]+\+[0-9]+$"), F.split(tc, r"\+").getItem(0).cast("int"))
         .alias("base_seconds"),
        F.when(tc.rlike(r"^[0-9]+$"), F.lit(0))
         .when(tc.rlike(r"^[0-9]+\+[0-9]+$"), F.split(tc, r"\+").getItem(1).cast("int"))
         .alias("increment_seconds"),
        F.col("g.end_time").cast("timestamp").alias("end_ts"),   # epoch -> UTC
        F.to_timestamp(
            F.concat_ws(
                " ",
                F.regexp_extract("g.pgn", r'\[UTCDate "([^"]+)"\]', 1),
                F.regexp_extract("g.pgn", r'\[UTCTime "([^"]+)"\]', 1),
            ),
            "yyyy.MM.dd HH:mm:ss",
        ).alias("start_ts"),                                     # UTC
        F.when(is_white, F.lit("white")).otherwise(F.lit("black")).alias("color"),
        my["rating"].cast("int").alias("my_rating"),
        opp["rating"].cast("int").alias("opp_rating"),
        F.lower(opp["username"]).alias("opp_username"),
        my["result"].alias("my_result_code"),
        opp["result"].alias("opp_result_code"),
        F.regexp_extract("g.pgn", r'\[ECO "([^"]+)"\]', 1).alias("eco_code"),
        F.col("g.eco").alias("eco_url"),
        F.col("g.pgn").alias("pgn"),
        "_source_file",
        "_ingested_at",
    )
    .withColumn(
        "result",
        F.when(F.col("my_result_code") == "win", "won")
         .when(F.col("my_result_code").isin(DRAW_CODES), "draw")
         .otherwise("lost"),
    )
    # How the game ended: my code explains a loss/draw, opponent's explains a win.
    .withColumn(
        "termination",
        F.when(F.col("result") == "won", F.col("opp_result_code"))
         .otherwise(F.col("my_result_code")),
    )
    .withColumn(
        "is_friend",
        F.col("opp_username").isin([f.lower() for f in FRIENDS]),
    )
    .withColumn(
        "is_self_play",
        F.col("opp_username").isin([a.lower() for a in ACCOUNTS]),
    )
    .withColumn("eco_code", F.when(F.col("eco_code") == "", None).otherwise(F.col("eco_code")))
)

# COMMAND ----------

# ---- silver.moves: one row per half-move, standard chess only ------------
# SAN pattern notes:
#   - matches "1. e4 {" and black's "1... c5 {" alike (both end ". san {")
#   - '=' is REQUIRED in the class: the 2024 pipeline's pattern lacked it,
#     so promotion moves like e8=Q+ silently never matched (fixed here)
SAN_PATTERN = r"\.\s([a-zA-Z0-9\-+#=]+)\s*\{"
CLK_PATTERN = r"\[%clk (\d+:\d+:\d+(?:\.\d+)?)\]"

moves_df = (
    games_df
    .filter(F.col("rules") == "chess")
    .select(
        "game_uuid", "account", "color",
        F.regexp_extract_all(F.col("pgn"), F.lit(SAN_PATTERN), F.lit(1)).alias("sans"),
        F.regexp_extract_all(F.col("pgn"), F.lit(CLK_PATTERN), F.lit(1)).alias("clks"),
    )
    .select(
        "game_uuid", "account", "color",
        F.posexplode(F.arrays_zip("sans", "clks")).alias("pos", "mv"),
    )
    .select(
        "game_uuid",
        "account",
        (F.col("pos") + 1).alias("ply"),
        (F.floor(F.col("pos") / 2) + 1).cast("int").alias("move_number"),
        F.when(F.col("pos") % 2 == 0, "white").otherwise("black").alias("side"),
        F.col("mv.sans").alias("san"),
        F.col("mv.clks").alias("clock_str"),
        (
            F.split("mv.clks", ":").getItem(0).cast("double") * 3600
            + F.split("mv.clks", ":").getItem(1).cast("double") * 60
            + F.split("mv.clks", ":").getItem(2).cast("double")
        ).alias("clock_seconds"),
        (F.when(F.col("pos") % 2 == 0, "white").otherwise("black") == F.col("color"))
            .alias("is_my_move"),
    )
)

# COMMAND ----------

# ---- MERGE upserts (create-on-first-run) ----------------------------------

def merge_into(df, table: str, key_cols: list[str]) -> None:
    """Upsert df into table on key_cols; create the table on first run."""
    if not spark.catalog.tableExists(table):
        df.write.saveAsTable(table)
        print(f"created {table}: {spark.table(table).count():,} rows")
        return
    cond = " AND ".join(f"t.{k} = s.{k}" for k in key_cols)
    (
        DeltaTable.forName(spark, table).alias("t")
        .merge(df.alias("s"), cond)
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
    print(f"merged into {table}: now {spark.table(table).count():,} rows")

merge_into(games_df, f"{CATALOG}.silver.games", ["game_uuid", "account"])
merge_into(moves_df, f"{CATALOG}.silver.moves", ["game_uuid", "account", "ply"])

# COMMAND ----------

# MAGIC %md ## Verification

# COMMAND ----------

g = f"{CATALOG}.silver.games"
m = f"{CATALOG}.silver.moves"

# 1. Dedup proof: bronze rows in, unique games out.
display(spark.sql(f"""
  SELECT
    (SELECT count(*) FROM (SELECT explode(games) FROM {CATALOG}.bronze.games_raw))
      AS bronze_game_rows_incl_dupes,
    count(*)                                   AS silver_games,
    count(DISTINCT game_uuid, account)         AS distinct_keys,
    sum(CASE WHEN is_self_play THEN 1 ELSE 0 END) AS self_play_games,
    sum(CASE WHEN is_friend THEN 1 ELSE 0 END)    AS friend_games
  FROM {g}
"""))

# 2. The time_class x time_control anomaly, on the record.
display(spark.sql(f"""
  SELECT account, time_class, base_seconds, count(*) AS games
  FROM {g} WHERE rules = 'chess'
  GROUP BY account, time_class, base_seconds
  ORDER BY account, time_class, base_seconds
"""))

# 3. Model target population (gold will filter to this).
display(spark.sql(f"""
  SELECT count(*) AS model_population
  FROM {g}
  WHERE rated AND rules = 'chess' AND time_class = 'rapid'
    AND NOT is_friend AND NOT is_self_play
"""))

# 4. Moves: volume, promotion fix landed, and no orphans.
display(spark.sql(f"""
  SELECT
    count(*)                                    AS move_rows,
    sum(CASE WHEN san LIKE '%=%' THEN 1 ELSE 0 END) AS promotion_moves,
    count(DISTINCT game_uuid)                   AS games_with_moves
  FROM {m}
"""))
display(spark.sql(f"""
  SELECT count(*) AS orphan_move_rows
  FROM {m} mv LEFT ANTI JOIN {g} gm
    ON mv.game_uuid = gm.game_uuid AND mv.account = gm.account
"""))