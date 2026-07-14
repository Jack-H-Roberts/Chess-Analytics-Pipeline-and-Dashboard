# Databricks notebook source
# MAGIC %md
# MAGIC # 04_gold — the 48-feature game table + dashboard view
# MAGIC
# MAGIC **Row population:** model-target games only — rated, standard chess,
# MAGIC rapid, not vs. friends, not self-play.
# MAGIC
# MAGIC **Context population:** the person's ENTIRE activity timeline — both
# MAGIC accounts, all time classes and variants, rated or not. Fatigue, tilt,
# MAGIC and streak features describe the human, not one account's ladder.
# MAGIC
# MAGIC Every window looks strictly backward, so past rows never change when
# MAGIC new games arrive: gold is a deterministic projection of silver and is
# MAGIC rebuilt wholesale (overwrite) each run. MERGE-style dedup belongs to
# MAGIC silver, where duplicates actually occur.
# MAGIC
# MAGIC Feature names replicate the 2024 pipeline's 48 exactly (training-code
# MAGIC parity), with two behavioral fixes: castling detection now matches
# MAGIC `O-O+`/`O-O#`, and time-since-last uses the API's real end times.

# COMMAND ----------

from pyspark.sql import functions as F, Window

CATALOG = "chess"
GOLD_TABLE = f"{CATALOG}.gold.game_features"

DAY_S = 24 * 3600
WEEK_S = 7 * 24 * 3600

# The 2024 pipeline's ECO vocabulary: my nine common openings + Other.
ECO_CODES = ["A00", "A40", "A45", "B10", "B12", "B13", "D00", "D02", "D10"]

MODEL_POPULATION = (
    "rated AND rules = 'chess' AND time_class = 'rapid' "
    "AND NOT is_friend AND NOT is_self_play"
)

# COMMAND ----------

# ---- 1. Person-wide activity timeline with backward-looking windows ------
# One row per game EVER played (any account/class/variant). Bughouse lacks
# start_ts, so events order on coalesce(start_ts, end_ts).

games = spark.table(f"{CATALOG}.silver.games")

timeline = (
    games
    .withColumn("activity_ts", F.coalesce("start_ts", "end_ts"))
    .withColumn("activity_sec", F.unix_timestamp("activity_ts"))
    .withColumn("end_sec", F.unix_timestamp("end_ts"))
)

w_order = Window.orderBy("activity_sec", "game_uuid")
w_24h = Window.orderBy("activity_sec").rangeBetween(-(DAY_S - 1), -1)
w_7d = Window.orderBy("activity_sec").rangeBetween(-(WEEK_S - 1), -1)

def pct(result_value: str, w) -> "F.Column":
    """% of games in window w with the given result; 0.0 when window empty."""
    hits = F.sum(F.when(F.col("result") == result_value, 1).otherwise(0)).over(w)
    n = F.count("*").over(w)
    return F.when(n > 0, F.round(100.0 * hits / n, 2)).otherwise(F.lit(0.0))

timeline = (
    timeline
    .withColumn("GameOfDay",  (F.count("*").over(w_24h) + 1).cast("int"))
    .withColumn("GameOfWeek", (F.count("*").over(w_7d) + 1).cast("int"))
    .withColumn("DailyWinPerc",   pct("won",  w_24h))
    .withColumn("DailyLossPerc",  pct("lost", w_24h))
    .withColumn("DailyDrawPerc",  pct("draw", w_24h))
    .withColumn("WeeklyWinPerc",  pct("won",  w_7d))
    .withColumn("WeeklyLossPerc", pct("lost", w_7d))
    .withColumn("WeeklyDrawPerc", pct("draw", w_7d))
    .withColumn("last_result",  F.lag("result", 1).over(w_order))
    .withColumn("last2_result", F.lag("result", 2).over(w_order))
    # Real previous-game end time (the 2024 code approximated end as
    # start + my clock usage). Clamped at 0 for overlapping daily games.
    .withColumn("prev_end_sec", F.lag("end_sec", 1).over(w_order))
    .withColumn(
        "TimeSinceLast",
        F.when(
            F.col("prev_end_sec").isNotNull(),
            F.greatest(F.lit(0.0), (F.col("activity_sec") - F.col("prev_end_sec")).cast("double")),
        ),
    )
)

# COMMAND ----------

# ---- 2. Per-game move & castling aggregates from silver.moves ------------
# Regexes include the +/# suffixes the 2024 exact-string match missed.

SHORT = r"^O-O[+#]?$"
LONG = r"^O-O-O[+#]?$"

mv = spark.table(f"{CATALOG}.silver.moves")
is_castle = F.col("san").rlike(SHORT) | F.col("san").rlike(LONG)

move_aggs = (
    mv.groupBy("game_uuid", "account")
    .agg(
        F.sum(F.when(F.col("is_my_move"), 1).otherwise(0)).cast("int").alias("MyNumMoves"),
        F.sum(F.when(~F.col("is_my_move"), 1).otherwise(0)).cast("int").alias("OppNumMoves"),
        F.expr("max_by(CASE WHEN is_my_move THEN clock_seconds END, "
               "CASE WHEN is_my_move THEN ply END)").alias("my_last_clock"),
        F.expr("max_by(CASE WHEN NOT is_my_move THEN clock_seconds END, "
               "CASE WHEN NOT is_my_move THEN ply END)").alias("opp_last_clock"),
        F.max(F.when(F.col("is_my_move") & F.col("san").rlike(SHORT), 1).otherwise(0)).alias("ICastledShort"),
        F.max(F.when(F.col("is_my_move") & F.col("san").rlike(LONG), 1).otherwise(0)).alias("ICastledLong"),
        F.max(F.when(~F.col("is_my_move") & F.col("san").rlike(SHORT), 1).otherwise(0)).alias("OppCastledShort"),
        F.max(F.when(~F.col("is_my_move") & F.col("san").rlike(LONG), 1).otherwise(0)).alias("OppCastledLong"),
        F.min(F.when(F.col("is_my_move") & is_castle, F.col("ply"))).alias("my_castle_ply"),
        F.min(F.when(~F.col("is_my_move") & is_castle, F.col("ply"))).alias("opp_castle_ply"),
    )
)

# COMMAND ----------

# ---- 3. Assemble the 48 features for the model population ----------------

target = (
    timeline.filter(MODEL_POPULATION)
    .join(move_aggs, ["game_uuid", "account"], "left")
    .withColumn("local_ts", F.from_utc_timestamp("start_ts", "America/New_York"))
    .withColumn("day_name", F.date_format("local_ts", "EEEE"))
)

my_time = F.round(
    F.when(
        F.coalesce("MyNumMoves", F.lit(0)) > 0,
        F.col("base_seconds") + F.col("increment_seconds") * F.col("MyNumMoves") - F.col("my_last_clock"),
    ).otherwise(F.lit(0.0)),
    2,
)
opp_time = F.round(
    F.when(
        F.coalesce("OppNumMoves", F.lit(0)) > 0,
        F.col("base_seconds") + F.col("increment_seconds") * F.col("OppNumMoves") - F.col("opp_last_clock"),
    ).otherwise(F.lit(0.0)),
    2,
)

gold = target.select(
    # -- identifiers (not features): join key + chronological split key
    "game_uuid",
    "start_ts",
    # -- the 48 features, in the 2024 pipeline's order --------------------
    F.when(F.col("account") == "cosmos_iv", 0).otherwise(1).alias("Account"),
    *[
        F.when(F.col("day_name") == d, 1).otherwise(0).alias(f"Is{d}")
        for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    ],
    (
        F.hour("local_ts") * 3600 + F.minute("local_ts") * 60 + F.second("local_ts")
    ).cast("int").alias("TimeOfDay"),
    "GameOfDay",
    "GameOfWeek",
    F.col("base_seconds").alias("TimeControl"),
    (F.col("my_rating") - F.col("opp_rating")).cast("int").alias("EloDifference"),
    F.when(F.col("color") == "white", 0).otherwise(1).alias("Color"),
    F.when(
        F.col("my_castle_ply").isNotNull()
        & (F.col("opp_castle_ply").isNull() | (F.col("my_castle_ply") < F.col("opp_castle_ply"))),
        1,
    ).otherwise(0).alias("ICastledFirst"),
    F.coalesce("ICastledShort", F.lit(0)).alias("ICastledShort"),
    F.coalesce("ICastledLong", F.lit(0)).alias("ICastledLong"),
    F.coalesce("OppCastledShort", F.lit(0)).alias("OppCastledShort"),
    F.coalesce("OppCastledLong", F.lit(0)).alias("OppCastledLong"),
    F.when(F.col("last_result") == "won", 1).otherwise(0).alias("LastResultIsWin"),
    F.when(F.col("last_result") == "draw", 1).otherwise(0).alias("LastResultIsDraw"),
    F.when(F.col("last_result") == "lost", 1).otherwise(0).alias("LastResultIsLoss"),
    F.when(F.col("last2_result") == "won", 1).otherwise(0).alias("2ndLastResultIsWin"),
    F.when(F.col("last2_result") == "draw", 1).otherwise(0).alias("2ndLastResultIsDraw"),
    F.when(F.col("last2_result") == "lost", 1).otherwise(0).alias("2ndLastResultIsLoss"),
    F.coalesce("MyNumMoves", F.lit(0)).alias("MyNumMoves"),
    F.coalesce("OppNumMoves", F.lit(0)).alias("OppNumMoves"),
    my_time.alias("MyTotalTime"),
    opp_time.alias("OppTotalTime"),
    F.when(F.coalesce("MyNumMoves", F.lit(0)) > 0, F.round(my_time / F.col("MyNumMoves"), 2))
     .otherwise(F.lit(0.0)).alias("MyAvgTPM"),
    F.when(F.coalesce("OppNumMoves", F.lit(0)) > 0, F.round(opp_time / F.col("OppNumMoves"), 2))
     .otherwise(F.lit(0.0)).alias("OppAvgTPM"),
    "TimeSinceLast",
    "DailyWinPerc", "DailyLossPerc", "DailyDrawPerc",
    "WeeklyWinPerc", "WeeklyLossPerc", "WeeklyDrawPerc",
    *[F.when(F.col("eco_code") == e, 1).otherwise(0).alias(f"ECO_{e}") for e in ECO_CODES],
    F.when(F.col("eco_code").isin(ECO_CODES), 0).otherwise(1).alias("ECO_Other"),
    # -- target ------------------------------------------------------------
    F.when(F.col("result") == "won", 0)
     .when(F.col("result") == "draw", 1)
     .otherwise(2).alias("Result"),
)

gold.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(GOLD_TABLE)
print(f"rebuilt {GOLD_TABLE}: {spark.table(GOLD_TABLE).count():,} rows, "
      f"{len(spark.table(GOLD_TABLE).columns) - 3} features (+ 2 ids + target)")

# COMMAND ----------

# ---- 4. Dashboard view: all games, local time, population flag -----------

spark.sql(f"""
CREATE OR REPLACE VIEW {CATALOG}.gold.dashboard_games AS
SELECT
  *,
  from_utc_timestamp(coalesce(start_ts, end_ts), 'America/New_York') AS local_ts,
  hour(from_utc_timestamp(coalesce(start_ts, end_ts), 'America/New_York')) AS local_hour,
  date_format(from_utc_timestamp(coalesce(start_ts, end_ts), 'America/New_York'), 'EEEE') AS local_day,
  ({MODEL_POPULATION}) AS is_model_population
FROM {CATALOG}.silver.games
""")
print(f"created view {CATALOG}.gold.dashboard_games")

# COMMAND ----------

# MAGIC %md ## Verification

# COMMAND ----------

t = spark.table(GOLD_TABLE)

# Row count must equal the model population; exactly one null TimeSinceLast
# (the first game ever played); every game in exactly one ECO bucket.
display(spark.sql(f"""
  SELECT
    (SELECT count(*) FROM {CATALOG}.silver.games WHERE {MODEL_POPULATION}) AS model_population,
    count(*)                                        AS gold_rows,
    sum(CASE WHEN TimeSinceLast IS NULL THEN 1 ELSE 0 END) AS null_time_since_last,
    sum(ECO_A00+ECO_A40+ECO_A45+ECO_B10+ECO_B12+ECO_B13+ECO_D00+ECO_D02+ECO_D10+ECO_Other)
                                                    AS eco_onehot_sum
  FROM {GOLD_TABLE}
"""))

# Feature sanity: distributions that should look like chess.
display(spark.sql(f"""
  SELECT
    round(avg(GameOfDay), 2)      AS avg_game_of_day,
    max(GameOfDay)                AS max_game_of_day,
    round(avg(DailyWinPerc), 2)   AS avg_daily_win_pct,
    round(avg(MyNumMoves), 1)     AS avg_my_moves,
    round(avg(MyTotalTime), 1)    AS avg_my_seconds_used,
    round(avg(ICastledShort), 3)  AS castle_short_rate,
    round(avg(EloDifference), 2)  AS avg_elo_diff,
    round(100 * avg(CASE WHEN Result = 0 THEN 1.0 ELSE 0 END), 2) AS win_pct,
    round(100 * avg(CASE WHEN Result = 1 THEN 1.0 ELSE 0 END), 2) AS draw_pct
  FROM {GOLD_TABLE}
"""))
