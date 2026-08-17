#!/usr/bin/env python3
"""
Fetch and rank GitHub users who list Victoria Falls (or Vic Falls) as their location.

Updates README.md between <!-- START_LEADERBOARD --> and <!-- END_LEADERBOARD -->.
"""

from __future__ import annotations

import html
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Literal

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GITHUB_API = "https://api.github.com"
SEARCH_QUERIES = (
    'location:"Victoria Falls"',
    'location:"Vic Falls"',
)
README_PATH = "README.md"
INDEX_PATH = "index.html"
START_MARKER = "<!-- START_LEADERBOARD -->"
END_MARKER = "<!-- END_LEADERBOARD -->"
AVATAR_WIDTH = 40
BIO_MAX_LEN = 80
REQUEST_TIMEOUT = 30
MAX_RETRIES = 5
PER_PAGE = 100
USER_AGENT = "vic-falls-github-leaderboard/1.0"

SortKey = Literal["followers", "repos", "contributions"]
DEFAULT_SORT: SortKey = "followers"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GitHubUser:
    login: str
    html_url: str
    avatar_url: str
    name: str | None
    bio: str | None
    location: str | None
    followers: int
    public_repos: int
    public_gists: int

    @property
    def contributions_score(self) -> int:
        """Proxy contribution metric: repos weighted higher than gists, plus followers."""
        return (self.public_repos * 10) + (self.public_gists * 2) + self.followers

    @classmethod
    def from_api(cls, payload: dict[str, Any]) -> GitHubUser:
        return cls(
            login=payload["login"],
            html_url=payload.get("html_url") or f"https://github.com/{payload['login']}",
            avatar_url=payload.get("avatar_url") or "",
            name=payload.get("name"),
            bio=payload.get("bio"),
            location=payload.get("location"),
            followers=int(payload.get("followers") or 0),
            public_repos=int(payload.get("public_repos") or 0),
            public_gists=int(payload.get("public_gists") or 0),
        )


# ---------------------------------------------------------------------------
# HTTP client with rate-limit awareness
# ---------------------------------------------------------------------------


class GitHubClient:
    def __init__(self, token: str | None) -> None:
        self.session = requests.Session()
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": USER_AGENT,
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        else:
            logger.warning(
                "No GITHUB_TOKEN set — unauthenticated requests are heavily rate-limited."
            )
        self.session.headers.update(headers)

    def close(self) -> None:
        self.session.close()

    def get(self, url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any] | list[Any]:
        last_error: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            except requests.RequestException as exc:
                last_error = exc
                sleep_for = min(2**attempt, 60)
                logger.warning(
                    "Request failed (%s). Retry %d/%d in %ds…",
                    exc,
                    attempt,
                    MAX_RETRIES,
                    sleep_for,
                )
                time.sleep(sleep_for)
                continue

            if response.status_code == 403 and self._is_rate_limited(response):
                reset_at = response.headers.get("X-RateLimit-Reset")
                wait = self._seconds_until_reset(reset_at)
                logger.warning(
                    "Rate limited. Waiting %ds before retry %d/%d…",
                    wait,
                    attempt,
                    MAX_RETRIES,
                )
                time.sleep(wait)
                continue

            if response.status_code == 422:
                # Often means the search query returned no usable results or is invalid.
                logger.error("GitHub rejected the query (HTTP 422): %s", response.text[:300])
                return {"total_count": 0, "items": []}

            if response.status_code >= 500:
                sleep_for = min(2**attempt, 60)
                logger.warning(
                    "GitHub server error %s. Retry %d/%d in %ds…",
                    response.status_code,
                    attempt,
                    MAX_RETRIES,
                    sleep_for,
                )
                time.sleep(sleep_for)
                continue

            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                logger.error("HTTP error %s: %s", response.status_code, response.text[:500])
                raise SystemExit(1) from exc

            try:
                return response.json()
            except ValueError as exc:
                raise SystemExit("GitHub returned a non-JSON response.") from exc

        raise SystemExit(f"Failed after {MAX_RETRIES} retries: {last_error}")

    @staticmethod
    def _is_rate_limited(response: requests.Response) -> bool:
        remaining = response.headers.get("X-RateLimit-Remaining")
        if remaining == "0":
            return True
        body = (response.text or "").lower()
        return "rate limit" in body or "secondary rate limit" in body

    @staticmethod
    def _seconds_until_reset(reset_header: str | None) -> int:
        if not reset_header:
            return 60
        try:
            reset_ts = int(reset_header)
        except ValueError:
            return 60
        wait = reset_ts - int(time.time()) + 2
        return max(wait, 5)


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def search_user_logins(client: GitHubClient) -> list[str]:
    """Collect unique user logins matching Victoria Falls location queries."""
    seen: dict[str, None] = {}

    for query in SEARCH_QUERIES:
        page = 1
        while True:
            logger.info("Searching users: %s (page %d)", query, page)
            payload = client.get(
                f"{GITHUB_API}/search/users",
                params={"q": query, "per_page": PER_PAGE, "page": page},
            )

            if not isinstance(payload, dict):
                logger.error("Unexpected search response type: %s", type(payload).__name__)
                break

            items = payload.get("items") or []
            if not items:
                break

            for item in items:
                login = item.get("login")
                if login:
                    seen.setdefault(login, None)

            # GitHub Search API caps at 1000 results; stop early when exhausted.
            total = int(payload.get("total_count") or 0)
            if page * PER_PAGE >= min(total, 1000) or len(items) < PER_PAGE:
                break
            page += 1

    logins = list(seen.keys())
    logger.info("Found %d unique user(s) from search.", len(logins))
    return logins


def fetch_user_details(client: GitHubClient, logins: Iterable[str]) -> list[GitHubUser]:
    users: list[GitHubUser] = []
    for login in logins:
        logger.info("Fetching profile: %s", login)
        payload = client.get(f"{GITHUB_API}/users/{login}")
        if not isinstance(payload, dict) or "login" not in payload:
            logger.warning("Skipping incomplete profile for %s", login)
            continue
        users.append(GitHubUser.from_api(payload))
    return users


# ---------------------------------------------------------------------------
# Sorting & rendering
# ---------------------------------------------------------------------------


SORT_KEY_FNS: dict[SortKey, Callable[[GitHubUser], int]] = {
    "followers": lambda u: u.followers,
    "repos": lambda u: u.public_repos,
    "contributions": lambda u: u.contributions_score,
}


def sort_users(users: list[GitHubUser], sort_by: SortKey = DEFAULT_SORT) -> list[GitHubUser]:
    key_fn = SORT_KEY_FNS[sort_by]
    return sorted(users, key=lambda u: (key_fn(u), u.followers, u.public_repos), reverse=True)


def truncate_bio(bio: str | None, max_len: int = BIO_MAX_LEN) -> str:
    if not bio:
        return "—"
    cleaned = " ".join(bio.split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1].rstrip() + "…"


def escape_md_cell(text: str) -> str:
    """Escape pipe characters so Markdown tables stay intact."""
    return text.replace("|", "\\|").replace("\n", " ")


def render_leaderboard(users: list[GitHubUser], sort_by: SortKey) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    metric_label = {
        "followers": "followers",
        "repos": "public repositories",
        "contributions": "contribution score (repos × 10 + gists × 2 + followers)",
    }[sort_by]

    lines: list[str] = [
        f"_Last updated: **{now}** · Sorted by **{metric_label}**_",
        "",
    ]

    if not users:
        lines.extend(
            [
                "> No developers currently list Victoria Falls or Vic Falls in their "
                "GitHub profile location. Be the first — see **How to appear on this list** below.",
                "",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "| Rank | Avatar | Username | Followers | Repos | Bio | Profile |",
            "| :---: | :---: | --- | ---: | ---: | --- | --- |",
        ]
    )

    for rank, user in enumerate(users, start=1):
        safe_avatar = html.escape(user.avatar_url, quote=True)
        avatar = (
            f'<img src="{safe_avatar}" width="{AVATAR_WIDTH}" height="{AVATAR_WIDTH}" '
            f'alt="{html.escape(user.login)}"/>'
            if user.avatar_url
            else "—"
        )
        display_name = escape_md_cell(user.name) if user.name else "—"
        username = f"**[{escape_md_cell(user.login)}]({user.html_url})**"
        if user.name:
            username = f"{username}<br/>_{escape_md_cell(display_name)}_"
        bio = escape_md_cell(truncate_bio(user.bio))
        profile = f"[Open]({user.html_url})"

        lines.append(
            f"| {rank} | {avatar} | {username} | {user.followers} | "
            f"{user.public_repos} | {bio} | {profile} |"
        )

    lines.append("")
    return "\n".join(lines)


def update_readme(leaderboard_md: str, path: str = README_PATH) -> None:
    try:
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
    except FileNotFoundError as exc:
        raise SystemExit(f"README not found at {path!r}. Create it before running.") from exc

    if START_MARKER not in content or END_MARKER not in content:
        raise SystemExit(
            f"README must contain both {START_MARKER} and {END_MARKER} markers."
        )

    pattern = re.compile(
        re.escape(START_MARKER) + r".*?" + re.escape(END_MARKER),
        re.DOTALL,
    )
    replacement = f"{START_MARKER}\n{leaderboard_md.rstrip()}\n{END_MARKER}"
    updated, count = pattern.subn(replacement, content, count=1)
    if count != 1:
        raise SystemExit("Failed to inject leaderboard into README markers.")

    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(updated)
    logger.info("Updated %s", path)


def render_html_page(users: list[GitHubUser], sort_by: SortKey) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    metric_label = {
        "followers": "followers",
        "repos": "public repositories",
        "contributions": "contribution score",
    }[sort_by]

    rows: list[str] = []
    if users:
        for rank, user in enumerate(users, start=1):
            avatar = (
                f'<img class="avatar" src="{html.escape(user.avatar_url, quote=True)}" '
                f'width="{AVATAR_WIDTH}" height="{AVATAR_WIDTH}" '
                f'alt="{html.escape(user.login)}">'
                if user.avatar_url
                else ""
            )
            name_html = (
                f'<div class="name">{html.escape(user.name)}</div>' if user.name else ""
            )
            bio = html.escape(truncate_bio(user.bio))
            rows.append(
                "        <tr>\n"
                f"          <td class=\"rank\">{rank}</td>\n"
                f"          <td>{avatar}</td>\n"
                "          <td>\n"
                f'            <a class="login" href="{html.escape(user.html_url, quote=True)}">'
                f"{html.escape(user.login)}</a>\n"
                f"            {name_html}\n"
                "          </td>\n"
                f"          <td class=\"num\">{user.followers}</td>\n"
                f"          <td class=\"num\">{user.public_repos}</td>\n"
                f"          <td class=\"bio\">{bio}</td>\n"
                f'          <td><a class="btn" href="{html.escape(user.html_url, quote=True)}">Open</a></td>\n'
                "        </tr>"
            )
        table_body = "\n".join(rows)
        table = f"""    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Avatar</th>
            <th>Username</th>
            <th>Followers</th>
            <th>Repos</th>
            <th>Bio</th>
            <th>Profile</th>
          </tr>
        </thead>
        <tbody>
{table_body}
        </tbody>
      </table>
    </div>"""
    else:
        table = (
            '    <p class="empty">No developers currently list Victoria Falls or Vic Falls '
            "in their GitHub profile location.</p>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Top Active GitHub Users in Victoria Falls, Zimbabwe</title>
  <meta name="description" content="Leaderboard of the most active GitHub developers in Victoria Falls (Vic Falls), Zimbabwe. Ranked by public followers and updated daily.">
  <style>
    :root {{
      --bg: #07140f;
      --panel: #0f241c;
      --ink: #e8f5ee;
      --muted: #9cb5a8;
      --line: #1e3d31;
      --accent: #2d8c6e;
      --accent-2: #c9a227;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
      background: radial-gradient(1200px 500px at 10% -10%, #143528 0%, var(--bg) 55%);
      color: var(--ink);
      line-height: 1.5;
    }}
    .wrap {{ max-width: 1080px; margin: 0 auto; padding: 48px 20px 80px; }}
    h1 {{ font-size: clamp(1.8rem, 4vw, 2.6rem); margin: 0 0 8px; }}
    .lede {{ color: var(--muted); max-width: 720px; }}
    .meta {{ color: var(--accent-2); font-size: 0.92rem; margin: 24px 0; }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: var(--panel);
    }}
    table {{ width: 100%; border-collapse: collapse; min-width: 720px; }}
    th, td {{ padding: 12px 14px; text-align: left; border-bottom: 1px solid var(--line); vertical-align: middle; }}
    th {{ font-size: 0.78rem; letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted); }}
    tr:last-child td {{ border-bottom: 0; }}
    .rank, .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .avatar {{ border-radius: 50%; display: block; }}
    .login {{ color: #7ee0bc; text-decoration: none; font-weight: 650; }}
    .name {{ color: var(--muted); font-size: 0.85rem; }}
    .bio {{ color: var(--muted); max-width: 280px; }}
    .btn {{
      display: inline-block;
      color: var(--bg);
      background: var(--accent);
      text-decoration: none;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 0.85rem;
      font-weight: 650;
    }}
    .how {{ margin-top: 48px; color: var(--muted); }}
    .how a {{ color: #7ee0bc; }}
    .empty {{ color: var(--muted); }}
  </style>
</head>
<body>
  <main class="wrap">
    <h1>Top Active GitHub Users in Victoria Falls</h1>
    <p class="lede">An automated leaderboard of open-source developers who list Victoria Falls or Vic Falls on their GitHub profile. Ranked by {html.escape(metric_label)}.</p>
    <p class="meta">Last updated: {html.escape(now)} · {len(users)} developer(s)</p>
{table}
    <section class="how">
      <h2>How to appear on this list</h2>
      <ol>
        <li>Open your <a href="https://github.com/settings/profile">GitHub profile settings</a>.</li>
        <li>Set <strong>Location</strong> to <code>Victoria Falls</code> or <code>Vic Falls</code>.</li>
        <li>This page refreshes every day at midnight UTC.</li>
      </ol>
    </section>
  </main>
</body>
</html>
"""


def write_index(users: list[GitHubUser], sort_by: SortKey, path: str = INDEX_PATH) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(render_html_page(users, sort_by))
    logger.info("Updated %s", path)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def resolve_sort_key() -> SortKey:
    raw = os.environ.get("SORT_BY", DEFAULT_SORT).strip().lower()
    if raw not in SORT_KEY_FNS:
        logger.warning("Unknown SORT_BY=%r — falling back to %s", raw, DEFAULT_SORT)
        return DEFAULT_SORT
    return raw  # type: ignore[return-value]


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    sort_by = resolve_sort_key()
    client = GitHubClient(token)

    try:
        logins = search_user_logins(client)
        users = fetch_user_details(client, logins) if logins else []
        ranked = sort_users(users, sort_by=sort_by)
        leaderboard = render_leaderboard(ranked, sort_by=sort_by)
        update_readme(leaderboard)
        write_index(ranked, sort_by=sort_by)
        logger.info("Leaderboard ready: %d developer(s).", len(ranked))
    finally:
        client.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
