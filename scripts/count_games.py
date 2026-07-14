#!/usr/bin/env python3
"""Pre-build verification: count raw games across both Chess.com accounts.

Confirms the "3,500+ games" resume claim before any infrastructure exists,
and produces the breakdowns tonight's scoping decisions depend on:
  - rated rapid count  (model target population)
  - rated blitz count  (scope-expansion candidate)
  - friend games, split rated/unrated  (filter vs. flag decision)
  - rules != "chess"   (variants the old TimeControl filter couldn't see)

Stdlib only, Python 3.10+. Requests are serial with a small delay per
Chess.com API guidelines.

Run:  python3 count_games.py
"""

import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

ACCOUNTS = ["Cosmos_IV", "CosmosSolitarus"]

# The old repo's hardcoded exclusion list (pgn_to_csv.py) — now just a flag.
FRIENDS = {u.lower() for u in (
    "As7rixx", "TreYerT12358", "GravityRebel",
    "papadabear514", "zepthro", "Flippjc", "ripjawe",
)}

# Chess.com 403s anonymous/default user agents; must be descriptive + contact.
USER_AGENT = (
    "chess-data-pipeline/0.1 "
    "(pre-build game count; contact: jackhroberts02@gmail.com)"
)

RESUME_CLAIM = 3500
DELAY_S = 0.3
RETRIES = 3


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 403:
                sys.exit(
                    f"\n403 from {url}\n"
                    "Chess.com rejected the request — usually the User-Agent. "
                    "Keep it descriptive and include contact info."
                )
            if e.code == 429:
                wait = 2 * attempt
                print(f"\n  rate-limited (429); sleeping {wait}s", file=sys.stderr)
                time.sleep(wait)
                last_err = e
                continue
            last_err = e
            time.sleep(1)
        except Exception as e:  # timeouts, connection resets
            last_err = e
            time.sleep(1)
    sys.exit(f"\nGiving up on {url} after {RETRIES} attempts: {last_err}")


def month_of(archive_url: str) -> str:
    parts = archive_url.rsplit("/", 2)
    return f"{parts[-2]}/{parts[-1]}"


def counter_line(c: Counter, limit: int | None = None) -> str:
    items = c.most_common(limit)
    return " | ".join(f"{k} {v:,}" for k, v in items) if items else "(none)"


def tally_account(username: str) -> dict[str, Counter]:
    base = f"https://api.chess.com/pub/player/{username.lower()}/games/archives"
    archives = fetch_json(base).get("archives", [])
    tallies = {
        "flat": Counter(),          # total / rated / friend counts
        "time_class": Counter(),
        "rules": Counter(),
        "time_control": Counter(),
    }
    if not archives:
        print(f"\n{username}: no archives found")
        return tallies

    print(
        f"\n{username}: {len(archives)} monthly archives "
        f"({month_of(archives[0])} -> {month_of(archives[-1])})"
    )

    for i, url in enumerate(archives, 1):
        games = fetch_json(url).get("games", [])
        print(
            f"  [{i:>3}/{len(archives)}] {month_of(url)}: {len(games):>4} games",
            end="\r",
            flush=True,
        )
        for g in games:
            flat = tallies["flat"]
            tc_class = g.get("time_class", "?")
            rules = g.get("rules", "?")
            rated = bool(g.get("rated"))

            flat["total"] += 1
            flat["rated" if rated else "unrated"] += 1
            if rated and tc_class == "rapid":
                flat["rated_rapid"] += 1
            if rated and tc_class == "blitz":
                flat["rated_blitz"] += 1

            tallies["time_class"][tc_class] += 1
            tallies["rules"][rules] += 1
            tallies["time_control"][g.get("time_control", "?")] += 1

            white = g.get("white", {}).get("username", "").lower()
            black = g.get("black", {}).get("username", "").lower()
            if white in FRIENDS or black in FRIENDS:
                flat["friend"] += 1
                flat["friend_rated" if rated else "friend_unrated"] += 1
        time.sleep(DELAY_S)

    print(" " * 70, end="\r")  # clear progress line
    f = tallies["flat"]
    print(f"  total games ............. {f['total']:,}")
    print(f"  time_class .............. {counter_line(tallies['time_class'])}")
    print(f"  rules ................... {counter_line(tallies['rules'])}")
    print(f"  rated / unrated ......... {f['rated']:,} / {f['unrated']:,}")
    print(f"  rated rapid (model pop) . {f['rated_rapid']:,}")
    print(f"  rated blitz ............. {f['rated_blitz']:,}")
    print(
        f"  friend games ............ {f['friend']:,} "
        f"(rated {f['friend_rated']:,} / unrated {f['friend_unrated']:,})"
    )
    print(f"  top time_controls ....... {counter_line(tallies['time_control'], 6)}")
    return tallies


def main() -> None:
    grand: dict[str, Counter] = {
        "flat": Counter(),
        "time_class": Counter(),
        "rules": Counter(),
        "time_control": Counter(),
    }
    for username in ACCOUNTS:
        tallies = tally_account(username)
        for key, counter in tallies.items():
            grand[key].update(counter)

    f = grand["flat"]
    total = f["total"]
    margin = total - RESUME_CLAIM
    verdict = f"PASS (+{margin:,})" if margin >= 0 else f"FAIL ({margin:,})"
    non_chess = sum(v for k, v in grand["rules"].items() if k != "chess")

    print("\n" + "=" * 68)
    print(f"TOTAL RAW GAMES: {total:,}   ->   \"{RESUME_CLAIM:,}+\" claim: {verdict}")
    print(
        f"rated rapid {f['rated_rapid']:,} | rated blitz {f['rated_blitz']:,} | "
        f"friend {f['friend']:,} (rated {f['friend_rated']:,}) | "
        f"non-chess variants {non_chess:,}"
    )
    print("=" * 68)


if __name__ == "__main__":
    main()
