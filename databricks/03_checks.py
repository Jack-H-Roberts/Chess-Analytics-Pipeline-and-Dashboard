# Databricks notebook source
# MAGIC %md
# MAGIC # 03_checks — quality gates for the silver layer
# MAGIC
# MAGIC Runs every check, prints PASS/FAIL for each, and raises at the end if
# MAGIC anything failed — so one bad run shows the complete picture in the log.
# MAGIC The raise fails this task, which fails the Lakeflow job, which sends
# MAGIC the failure email. Schema is doubly enforced: Delta rejects
# MAGIC schema-violating writes at the source, and this notebook asserts the
# MAGIC contract columns explicitly.

# COMMAND ----------

CATALOG = "chess"

# Known floors. Archives only accumulate, so these may rise but never fall.
MIN_TOTAL_GAMES = 5_769        # silver row count at first full load
MIN_MODEL_POPULATION = 3_500   # the resume's headline floor

REQUIRED_GAME_COLS = {
    "game_uuid", "account", "url", "rated", "time_class", "rules",
    "time_control", "base_seconds", "increment_seconds",
    "end_ts", "start_ts", "color", "my_rating", "opp_rating",
    "opp_username", "result", "termination", "eco_code",
    "is_friend", "is_self_play",
}
REQUIRED_MOVE_COLS = {
    "game_uuid", "account", "ply", "move_number", "side",
    "san", "clock_seconds", "is_my_move",
}

failures: list[str] = []

def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(f"{name} ({detail})")

# COMMAND ----------

games = spark.table(f"{CATALOG}.silver.games")
moves = spark.table(f"{CATALOG}.silver.moves")

# --- schema contracts ------------------------------------------------------
check("games schema contract", REQUIRED_GAME_COLS <= set(games.columns),
      f"missing: {sorted(REQUIRED_GAME_COLS - set(games.columns)) or 'none'}")
check("moves schema contract", REQUIRED_MOVE_COLS <= set(moves.columns),
      f"missing: {sorted(REQUIRED_MOVE_COLS - set(moves.columns)) or 'none'}")

# --- uniqueness (the MERGE keys hold) --------------------------------------
n_games = games.count()
n_game_keys = games.select("game_uuid", "account").distinct().count()
check("games key uniqueness", n_games == n_game_keys,
      f"{n_games:,} rows vs {n_game_keys:,} keys")

n_moves = moves.count()
n_move_keys = moves.select("game_uuid", "account", "ply").distinct().count()
check("moves key uniqueness", n_moves == n_move_keys,
      f"{n_moves:,} rows vs {n_move_keys:,} keys")

# --- row-count floors -------------------------------------------------------
check("games row-count floor", n_games >= MIN_TOTAL_GAMES,
      f"{n_games:,} >= {MIN_TOTAL_GAMES:,}")

model_pop = games.filter(
    "rated AND rules = 'chess' AND time_class = 'rapid' "
    "AND NOT is_friend AND NOT is_self_play"
).count()
check("model population floor", model_pop >= MIN_MODEL_POPULATION,
      f"{model_pop:,} >= {MIN_MODEL_POPULATION:,}")

# --- bronze -> silver reconciliation ---------------------------------------
bronze_keys = spark.sql(f"""
    SELECT count(DISTINCT coalesce(g.uuid, g.url),
                  regexp_extract(_source_file, 'username=([^/]+)', 1)) AS k
    FROM {CATALOG}.bronze.games_raw
    LATERAL VIEW explode(games) exploded AS g
""").first()["k"]
check("bronze->silver reconciliation", bronze_keys == n_games,
      f"bronze distinct keys {bronze_keys:,} vs silver rows {n_games:,}")

# --- referential + domain invariants ---------------------------------------
orphans = moves.join(games, ["game_uuid", "account"], "left_anti").count()
check("no orphan move rows", orphans == 0, f"{orphans:,} orphans")

null_start_chess = games.filter("rules = 'chess' AND start_ts IS NULL").count()
check("standard-chess games all have start_ts", null_start_chess == 0,
      f"{null_start_chess:,} null (variants may be null; chess may not)")

bad_result = games.filter("result NOT IN ('won','draw','lost')").count()
check("result domain", bad_result == 0, f"{bad_result:,} out-of-domain")

# COMMAND ----------

if failures:
    raise AssertionError(
        f"{len(failures)} quality check(s) failed: " + " | ".join(failures)
    )
print(f"\nAll checks passed — {n_games:,} games, {n_moves:,} moves, "
      f"model population {model_pop:,}.")
