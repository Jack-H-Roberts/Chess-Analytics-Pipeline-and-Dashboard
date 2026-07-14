"""Scheduled ingestion: Chess.com monthly archives -> S3 raw layer.

Runs as an AWS Lambda on an EventBridge schedule (monthly) and is safe to
invoke manually at any time. One code path covers backfill and incremental:

  1. Ask the Chess.com API which monthly archives exist per account.
  2. List which (username, year, month) partitions already exist in S3.
  3. Fetch every archive that is missing, plus the TRAILING_MONTHS most
     recent archives regardless -- refreshing the in-progress month and
     catching the just-closed month's final games after rollover.
  4. Land each archive's JSON exactly as received at
       raw/chesscom/games/username=<u>/year=<yyyy>/month=<mm>/games_<utc-timestamp>.json
     (Hive-style partitions, which Databricks reads as partition columns).

The first invocation against an empty bucket is therefore the full
backfill. A failed month is logged and skipped; because it never landed
in S3, the next run retries it automatically. Re-pulling a month is
harmless: the silver layer deduplicates on game uuid via MERGE.

No secrets: S3 access comes from the function's IAM role, and the
Chess.com API is public (it only requires a descriptive User-Agent).

Configuration (Lambda environment variables):
  BUCKET_NAME      target S3 bucket (required)
  USERNAMES        comma-separated Chess.com usernames (required)
  CONTACT_EMAIL    contact for the User-Agent string (required by
                   Chess.com's API guidelines)
  RAW_PREFIX       key prefix, default "raw/chesscom/games"
  TRAILING_MONTHS  recent archives to always re-pull, default 2
"""

import json
import logging
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")

BUCKET_NAME = os.environ["BUCKET_NAME"]
USERNAMES = [
    u.strip().lower() for u in os.environ.get("USERNAMES", "").split(",") if u.strip()
]
RAW_PREFIX = os.environ.get("RAW_PREFIX", "raw/chesscom/games")
TRAILING_MONTHS = int(os.environ.get("TRAILING_MONTHS", "2"))
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "")

USER_AGENT = f"chess-data-pipeline/1.0 (scheduled ingestion; contact: {CONTACT_EMAIL})"
RETRIES = 3
DELAY_S = 0.2


def fetch_url(url: str) -> bytes:
    """GET with the descriptive User-Agent Chess.com requires.

    Retries transient failures and 429 rate limits with backoff. A 403
    means the User-Agent was rejected -- fail immediately and loudly.
    """
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 403:
                raise RuntimeError(
                    f"403 from {url} -- Chess.com rejected the request "
                    "(check the User-Agent / CONTACT_EMAIL env var)"
                ) from e
            if e.code == 429:
                wait = 2 * attempt
                logger.warning("429 from %s; backing off %ss", url, wait)
                time.sleep(wait)
                last_err = e
                continue
            last_err = e
            time.sleep(1)
        except Exception as e:  # timeouts, connection resets
            last_err = e
            time.sleep(1)
    raise RuntimeError(f"Failed to fetch {url} after {RETRIES} attempts: {last_err}")


def archive_year_month(archive_url: str) -> tuple[str, str]:
    """'.../games/2026/07' -> ('2026', '07')."""
    parts = archive_url.rstrip("/").rsplit("/", 2)
    return parts[-2], parts[-1]


def landed_months(username: str) -> set[tuple[str, str]]:
    """Return the (year, month) partitions already present in S3."""
    prefix = f"{RAW_PREFIX}/username={username}/"
    found: set[tuple[str, str]] = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        for obj in page.get("Contents", []):
            segments = dict(
                seg.split("=", 1) for seg in obj["Key"].split("/") if "=" in seg
            )
            if "year" in segments and "month" in segments:
                found.add((segments["year"], segments["month"]))
    return found


def land_archive(username: str, archive_url: str) -> int:
    """Fetch one monthly archive and write it raw to its partition.

    The body is stored byte-for-byte as received: bronze preserves the
    source's truth, and all typing/cleaning happens downstream in silver.
    Returns the number of games in the archive (for logging/summary).
    """
    raw = fetch_url(archive_url)
    game_count = len(json.loads(raw).get("games", []))
    year, month = archive_year_month(archive_url)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    key = f"{RAW_PREFIX}/username={username}/year={year}/month={month}/games_{stamp}.json"
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=raw,
        ContentType="application/json",
        Metadata={
            "source-url": archive_url,
            "fetched-at": datetime.now(timezone.utc).isoformat(),
            "game-count": str(game_count),
        },
    )
    logger.info(
        "landed %s (%d games) -> s3://%s/%s", archive_url, game_count, BUCKET_NAME, key
    )
    return game_count


def sync_account(username: str) -> dict:
    """Diff the API's archive list against S3 and land whatever is due."""
    archives_url = f"https://api.chess.com/pub/player/{username}/games/archives"
    archives = json.loads(fetch_url(archives_url)).get("archives", [])
    existing = landed_months(username)
    trailing = {archive_year_month(url) for url in archives[-TRAILING_MONTHS:]}

    summary = {
        "username": username,
        "archives_at_source": len(archives),
        "fetched": [],
        "skipped": 0,
        "games_fetched": 0,
        "errors": [],
    }
    for url in archives:
        ym = archive_year_month(url)
        if ym in existing and ym not in trailing:
            summary["skipped"] += 1
            continue
        try:
            summary["games_fetched"] += land_archive(username, url)
            summary["fetched"].append("/".join(ym))
        except Exception as e:
            logger.error("failed on %s: %s", url, e)
            summary["errors"].append(f"{'/'.join(ym)}: {e}")
        time.sleep(DELAY_S)
    return summary


def lambda_handler(event, context):
    if not USERNAMES:
        raise RuntimeError("USERNAMES environment variable is not set")
    if not CONTACT_EMAIL:
        raise RuntimeError("CONTACT_EMAIL environment variable is not set")

    results = [sync_account(u) for u in USERNAMES]
    errors = [err for r in results for err in r["errors"]]
    logger.info("run summary: %s", json.dumps(results))

    if errors:
        # Fail loudly so the run is visibly marked failed. Missing months
        # self-heal: they never landed, so the next run fetches them.
        raise RuntimeError(f"{len(errors)} archive(s) failed: {errors}")
    return {"ok": True, "results": results}
