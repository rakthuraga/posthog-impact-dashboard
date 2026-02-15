import os
import time
import math
import datetime as dt
from typing import Dict, List, Any, Tuple

import requests
import pandas as pd
import streamlit as st

OWNER = "PostHog"
REPO = "posthog"
FULL_REPO = f"{OWNER}/{REPO}"

DEFAULT_DAYS = 90
DEFAULT_CAP_PRS = 400
BATCH_SIZE = 25  # GraphQL aliases per request (safe + simple)


# -----------------------------
# GitHub API helpers
# -----------------------------
def gh_headers(token: str) -> Dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "impact-dashboard-streamlit",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"  # capital B required by GitHub
    return headers


def sleep_backoff(attempt: int) -> None:
    time.sleep(min(30, 2 ** attempt))


def rest_get(url: str, token: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    headers = gh_headers(token)

    for attempt in range(4):
        r = requests.get(url, headers=headers, params=params, timeout=30)

        # If it's a rate limit / secondary rate limit, backoff then retry
        if r.status_code in (403, 429):
            txt = (r.text or "").lower()
            is_rl = "rate limit" in txt or "secondary rate limit" in txt

            if is_rl and attempt < 3:
                sleep_backoff(attempt)
                continue

            # otherwise, raise a helpful error with headers
            rem = r.headers.get("X-RateLimit-Remaining")
            reset = r.headers.get("X-RateLimit-Reset")
            used = r.headers.get("X-RateLimit-Used")
            msg = ""
            try:
                msg = r.json().get("message", "")
            except Exception:
                msg = r.text[:200]

            raise RuntimeError(
                f"REST {r.status_code}. message={msg} remaining={rem} used={used} reset={reset} token_present={bool(token)}"
            )

        r.raise_for_status()
        return r.json()

    r.raise_for_status()


def get_core_remaining(token: str) -> int:
    data = rest_get("https://api.github.com/rate_limit", token=token)
    return int(data["resources"]["core"]["remaining"])


def graphql(query: str, variables: Dict[str, Any], token: str) -> Dict[str, Any]:
    headers = gh_headers(token)
    url = "https://api.github.com/graphql"
    for attempt in range(4):
        r = requests.post(url, headers=headers, json={"query": query, "variables": variables}, timeout=30)
        if r.status_code in (429, 403) and "rate limit" in r.text.lower():
            sleep_backoff(attempt)
            continue
        r.raise_for_status()
        data = r.json()
        if "errors" in data:
            raise RuntimeError(data["errors"])
        return data
    if r.status_code == 403:
        raise RuntimeError(f"GitHub GraphQL 403: {r.text[:400]}")
    r.raise_for_status()


def graphql_rate_limit(token: str) -> Dict[str, Any]:
    q = "query { rateLimit { remaining resetAt cost } }"
    return graphql(q, {}, token=token)["data"]["rateLimit"]


# -----------------------------
# Data fetch (bounded + PR-only)
# -----------------------------
def search_merged_pr_numbers(days: int, cap: int, token: str) -> List[int]:
    since = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    q = f"is:pr repo:{FULL_REPO} is:merged merged:>={since}"
    url = "https://api.github.com/search/issues"

    nums: List[int] = []
    page = 1
    while len(nums) < cap:
        data = rest_get(
            url,
            token=token,
            params={"q": q, "sort": "updated", "order": "desc", "per_page": 100, "page": page},
        )
        items = data.get("items", [])
        if not items:
            break
        nums.extend([it["number"] for it in items])
        page += 1

    return nums[:cap]


def fetch_recent_merged_prs_rest(days: int, cap: int, token: str) -> List[Dict[str, Any]]:
    """Cheap REST fallback: list /pulls (up to 10 pages) to collect enough merged PRs."""
    cutoff = dt.datetime.utcnow() - dt.timedelta(days=days)
    prs: List[Dict[str, Any]] = []
    page = 1

    while len(prs) < cap and page <= 10:  # more pages so we get enough merged PRs
        items = rest_get(
            f"https://api.github.com/repos/{FULL_REPO}/pulls",
            token=token,
            params={"state": "closed", "sort": "created", "direction": "desc", "per_page": 100, "page": page},
        )

        if not items:
            break

        for pr in items:
            if not pr.get("merged_at"):
                continue
            merged_at = pd.to_datetime(pr["merged_at"], utc=True)
            if merged_at.to_pydatetime().replace(tzinfo=None) < cutoff:
                continue

            prs.append(
                {
                    "number": pr.get("number"),
                    "url": pr.get("html_url"),
                    "title": pr.get("title"),
                    "author": {"login": (pr.get("user") or {}).get("login")},
                    "createdAt": pr.get("created_at"),
                    "mergedAt": pr.get("merged_at"),
                    "additions": 0,
                    "deletions": 0,
                    "changedFiles": 0,
                    "reviews": {"nodes": []},
                }
            )

            if len(prs) >= cap:
                break

        page += 1

    return prs


def fetch_pr_batch(pr_numbers: List[int], token: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    # PR-only fields (no issue events, no status checks)
    pr_fields = """
      number
      url
      title
      author { login }
      createdAt
      mergedAt
      additions
      deletions
      changedFiles
      reviews(first: 100) { nodes { author { login } state createdAt } }
    """

    pr_blocks = "\n".join(
        [f'pr{i}: pullRequest(number: {n}) {{ {pr_fields} }}' for i, n in enumerate(pr_numbers, 1)]
    )

    query = f"""
    query($owner:String!, $name:String!) {{
      rateLimit {{ remaining resetAt cost }}
      repository(owner:$owner, name:$name) {{
        {pr_blocks}
      }}
    }}
    """

    data = graphql(query, {"owner": OWNER, "name": REPO}, token=token)
    rl = data["data"]["rateLimit"]
    repo = data["data"]["repository"]

    prs = []
    for k, v in repo.items():
        if k.startswith("pr") and v is not None:
            prs.append(v)

    return rl, prs


def fetch_pr_details(pr_numbers: List[int], token: str, token_present: bool) -> List[Dict[str, Any]]:
    if not token_present:
        return []

    all_prs: List[Dict[str, Any]] = []
    i = 0
    remaining = 5000  # optimistic start; updated from rl2 after each batch

    while i < len(pr_numbers):
        if remaining < 200:
            break

        batch = 25 if remaining > 1000 else 10 if remaining > 400 else 5
        chunk = pr_numbers[i : i + batch]

        try:
            rl2, prs = fetch_pr_batch(chunk, token=token)
            remaining = rl2.get("remaining", remaining)
            all_prs.extend(prs)
        except Exception as e:
            if "rate limit" in str(e).lower() or "secondary" in str(e).lower():
                break
            raise

        i += batch

    return all_prs


# -----------------------------
# Transform + scoring
# -----------------------------
def safe_login(obj: Any) -> str:
    if isinstance(obj, dict) and obj.get("login"):
        return obj["login"]
    if isinstance(obj, str) and obj:
        return obj
    return "unknown"


def minmax(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - lo) / (hi - lo)


def build_pr_df(prs: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for pr in prs:
        author = safe_login(pr.get("author"))
        created_at = pd.to_datetime(pr.get("createdAt"), utc=True)
        merged_at = pd.to_datetime(pr.get("mergedAt"), utc=True)

        merge_hours = None
        if pd.notna(created_at) and pd.notna(merged_at):
            merge_hours = (merged_at - created_at).total_seconds() / 3600.0
            merge_hours = min(merge_hours, 24 * 14)  # cap at 2 weeks

        additions = pr.get("additions", 0) or 0
        deletions = pr.get("deletions", 0) or 0
        lines_changed = additions + deletions
        changed_files = pr.get("changedFiles", 0) or 0

        reviews = pr.get("reviews", {}).get("nodes", []) or []
        review_authors = [safe_login(r.get("author")) for r in reviews if r.get("author")]
        review_authors = [a for a in review_authors if a != "unknown"]

        rows.append(
            {
                "number": pr.get("number"),
                "url": pr.get("url"),
                "title": pr.get("title"),
                "author": author,
                "created_at": created_at,
                "merged_at": merged_at,
                "merge_hours": merge_hours,
                "additions": additions,
                "deletions": deletions,
                "lines_changed": lines_changed,
                "changed_files": changed_files,
                "review_authors": review_authors,  # list
            }
        )
    return pd.DataFrame(rows)


def compute_engineer_df(pr_df: pd.DataFrame) -> pd.DataFrame:
    df = pr_df.copy()

    # Dampen PR size dominance + trivial PR spam
    df["log_lines"] = df["lines_changed"].apply(lambda x: math.log1p(max(0, int(x))))
    df["log_files"] = df["changed_files"].apply(lambda x: math.log1p(max(0, int(x))))

    # Delivery & leverage aggregated on PR authorship
    author_agg = df.groupby("author").agg(
        prs_merged=("number", "count"),
        delivery=("log_lines", "sum"),
        leverage=("log_files", "sum"),
        median_merge_hours=("merge_hours", "median"),
        avg_merge_hours=("merge_hours", "mean"),
    ).reset_index()

    # Review contribution: count reviews WRITTEN by each reviewer across all PRs
    # explode review authors
    exploded = df[["number", "review_authors"]].explode("review_authors").dropna()
    exploded = exploded[exploded["review_authors"] != "unknown"]
    exploded = exploded[~exploded["review_authors"].astype(str).str.endswith("[bot]")]
    exploded = exploded.drop_duplicates(subset=["number", "review_authors"])
    if len(exploded) == 0:
        review_counts = pd.DataFrame({"author": [], "reviews_written": []})
    else:
        review_counts = (
            exploded.groupby("review_authors")
            .size()
            .reset_index(name="reviews_written")
            .rename(columns={"review_authors": "author"})
        )

    eng = author_agg.merge(review_counts, on="author", how="left")
    eng["reviews_written"] = eng["reviews_written"].fillna(0).astype(int)

    # Cycle time score: invert (lower merge time => higher score), then normalize
    # Using median is more robust than mean in small windows.
    eng["cycle_score_raw"] = eng["median_merge_hours"].apply(
        lambda h: 0.0 if pd.isna(h) else (1.0 / (1.0 + max(0.0, float(h))))
    )

    # Normalize to 0–1 for compositing
    eng["delivery_n"] = minmax(eng["delivery"])
    eng["reviews_n"] = minmax(eng["reviews_written"])
    eng["cycle_n"] = minmax(eng["cycle_score_raw"])
    eng["leverage_n"] = minmax(eng["leverage"])


    eng = eng[eng["author"] != "unknown"].reset_index(drop=True)

    return eng


def apply_weights(eng: pd.DataFrame, w: Dict[str, float], include_leverage: bool) -> pd.DataFrame:
    eng = eng.copy()

    if include_leverage:
        eng["impact_score"] = (
            w["delivery"] * eng["delivery_n"]
            + w["reviews"] * eng["reviews_n"]
            + w["cycle"] * eng["cycle_n"]
            + w["leverage"] * eng["leverage_n"]
        )
        eng["c_delivery"] = w["delivery"] * eng["delivery_n"]
        eng["c_reviews"] = w["reviews"] * eng["reviews_n"]
        eng["c_cycle"] = w["cycle"] * eng["cycle_n"]
        eng["c_leverage"] = w["leverage"] * eng["leverage_n"]
    else:
        eng["impact_score"] = (
            w["delivery"] * eng["delivery_n"]
            + w["reviews"] * eng["reviews_n"]
            + w["cycle"] * eng["cycle_n"]
        )
        eng["c_delivery"] = w["delivery"] * eng["delivery_n"]
        eng["c_reviews"] = w["reviews"] * eng["reviews_n"]
        eng["c_cycle"] = w["cycle"] * eng["cycle_n"]
        eng["c_leverage"] = 0.0

    return eng.sort_values("impact_score", ascending=False)


# -----------------------------
# Load (cached only for limited mode so token changes don't hit stale cache)
# -----------------------------
def _load(
    days: int, cap_prs: int, token: str, token_present: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if token_present:
        # keep search from burning limits (Search API has its own throttles)
        cap_prs = min(cap_prs, 400)
        pr_numbers = search_merged_pr_numbers(days=days, cap=cap_prs, token=token)

        # slice based on GraphQL budget so high cap doesn't cause secondary rate limit
        rl = graphql_rate_limit(token)
        remaining = rl.get("remaining", 0)
        max_prs = 250 if remaining > 1000 else 120 if remaining > 400 else 60
        pr_numbers = pr_numbers[:max_prs]

        prs = fetch_pr_details(pr_numbers, token=token, token_present=token_present)  # GraphQL path
    else:
        # REST-only limited mode: list endpoint, cap 50 to avoid anonymous limit
        prs = fetch_recent_merged_prs_rest(days=int(days), cap=min(50, int(cap_prs)), token=token)

    pr_df = build_pr_df(prs)
    eng_df = compute_engineer_df(pr_df)
    return pr_df, eng_df


@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_cached(days: int, cap_prs: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cached only for limited mode (no token). Token mode uses _load() so cache stays predictable."""
    return _load(days, cap_prs, "", False)


# -----------------------------
# UI
# -----------------------------
def main():
    st.set_page_config(page_title="PostHog Impact Dashboard (PR-only MVP)", layout="wide")
    st.title("Engineering Impact Dashboard — PostHog (PR-only MVP)")
    st.caption(
        "MVP in 1 hour: uses PR metadata + PR reviews only. No incident/revert linkage, no NLP, no identity reconciliation."
    )

    with st.sidebar:
        st.subheader("Authentication")

        token = st.text_input(
            "GitHub Token (recommended)",
            value="",
            type="password",
            help="Use a fine-grained token with read access to public repositories.",
        ).strip()

        token_provided = bool(token)
        st.caption(f"Token provided: {token_provided}")

        if st.button("Test token"):
            if not token:
                st.warning("Paste a token first.")
            else:
                try:
                    me = graphql("query { viewer { login } rateLimit { remaining } }", {}, token=token)
                    login = me["data"]["viewer"]["login"]
                    remaining = me["data"]["rateLimit"]["remaining"]
                    st.success(f"Token works. viewer={login}, remaining={remaining}")
                except Exception as e:
                    st.error(f"Token test failed: {e}")

        if st.button("Test REST token"):
            try:
                data = rest_get("https://api.github.com/rate_limit", token=token)
                core = data["resources"]["core"]
                st.success(f"REST ok. remaining={core['remaining']} reset={core['reset']}")
            except Exception as e:
                st.error(f"REST test failed: {e}")

        if st.button("Show current rate limit"):
            try:
                data = rest_get("https://api.github.com/rate_limit", token=token)
                core = data["resources"]["core"]
                reset_dt = dt.datetime.fromtimestamp(core["reset"])
                st.info(f"Core remaining={core['remaining']} resets_at={reset_dt} (local machine time)")
            except Exception as e:
                st.error(f"Rate limit check failed: {e}")

        if not token_provided:
            st.warning("Running in limited mode (no GitHub token): using REST-only and a small sample (up to 50 PRs). Paste a token for full results.")

        st.divider()
        st.subheader("Scope")
        safe_default_days = 30 if not token_provided else DEFAULT_DAYS
        safe_default_cap = 50 if not token_provided else DEFAULT_CAP_PRS
        days = st.number_input("Lookback (days)", min_value=14, max_value=365, value=safe_default_days, step=7)
        cap_prs = st.number_input("Max merged PRs", min_value=50, max_value=2000, value=safe_default_cap, step=25)

        st.divider()
        st.subheader("Metrics")
        if token_provided:
            include_leverage = st.toggle("Include Leverage (changed files)", value=True)
        else:
            include_leverage = False  # limited mode: cycle time only

        st.divider()
        st.subheader("Weights (auto-normalized)")
        if token_provided:
            w_delivery = st.slider("Delivery (log lines)", 0.0, 1.0, 0.40, 0.05)
            w_reviews  = st.slider("Reviews written",     0.0, 1.0, 0.25, 0.05)
            w_cycle    = st.slider("Cycle time (faster)", 0.0, 1.0, 0.20, 0.05)
            w_leverage = st.slider("Leverage (log files)",0.0, 1.0, 0.15, 0.05) if include_leverage else 0.0
            w_sum = w_delivery + w_reviews + w_cycle + (w_leverage if include_leverage else 0.0)
            if w_sum == 0:
                w_sum = 1.0
            weights = {
                "delivery": w_delivery / w_sum,
                "reviews": w_reviews / w_sum,
                "cycle": w_cycle / w_sum,
                "leverage": (w_leverage / w_sum) if include_leverage else 0.0,
            }
        else:
            st.caption("Limited mode: cycle time only.")
            weights = {"delivery": 0.0, "reviews": 0.0, "cycle": 1.0, "leverage": 0.0}

        st.caption(f"Normalized weights: {weights}")

        st.divider()
        if not token_provided:
            st.caption("Tip: Add a GitHub token to increase rate limits and fetch more PRs.")

    with st.spinner("Fetching merged PRs + reviews and computing scores..."):
        if token_provided:
            pr_df, eng_df = _load(int(days), int(cap_prs), token, True)
        else:
            pr_df, eng_df = load_cached(int(days), int(cap_prs))
        if pr_df.empty or eng_df.empty:
            st.error("No PR data returned. Double-check your GITHUB_TOKEN and try a smaller cap (e.g., 150) or window (e.g., 30 days).")
            st.stop()

    if not token_provided:
        st.info(
            "Limited mode: metrics below are based on cycle time and PR count only. "
            "Delivery, reviews, and leverage are empty. Add a token for full metrics."
        )

    scored = apply_weights(eng_df, weights, include_leverage=include_leverage)

    # Top 5 table
    st.subheader("Top 5 Impact Signals (conversation starter, not performance ranking)")
    top5 = scored.head(5).copy()

    cols = [
        "author",
        "impact_score",
        "prs_merged",
        "delivery",
        "reviews_written",
        "median_merge_hours",
    ]
    if include_leverage:
        cols.append("leverage")

    top5_disp = top5[cols].rename(columns={
        "author": "Engineer",
        "impact_score": "Impact Score",
        "prs_merged": "Merged PRs",
        "delivery": "Delivery (Σ log lines)",
        "reviews_written": "Reviews Written",
        "median_merge_hours": "Median Merge Time (hrs)",
        "leverage": "Leverage (Σ log files)",
    })

    st.dataframe(
        top5_disp,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Impact Score": st.column_config.NumberColumn(format="%.3f"),
            "Median Merge Time (hrs)": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    # One breakdown viz: contributions for selected engineer
    st.subheader("Breakdown")
    engineer = st.selectbox("Select engineer", options=scored["author"].tolist(), index=0)
    row = scored[scored["author"] == engineer].iloc[0]

    breakdown_rows = [
        ("Delivery", float(row["c_delivery"])),
        ("Reviews", float(row["c_reviews"])),
        ("Cycle time", float(row["c_cycle"])),
    ]
    if include_leverage:
        breakdown_rows.append(("Leverage", float(row["c_leverage"])))

    breakdown = pd.DataFrame(breakdown_rows, columns=["dimension", "contribution"]).set_index("dimension")
    st.bar_chart(breakdown, height=220)

    # Optional: quick PR drilldown (still PR-only)
    with st.expander("Show merged PRs for selected engineer (latest 25)"):
        prs_for = pr_df[pr_df["author"] == engineer].sort_values("merged_at", ascending=False).head(25)
        if token_provided:
            prs_for = prs_for[["number", "title", "lines_changed", "changed_files", "merge_hours", "url"]]
        else:
            prs_for = prs_for[["number", "title", "merge_hours", "url"]]
        st.dataframe(prs_for, use_container_width=True, hide_index=True)

    st.caption(
        "Limitations: PR-only MVP. Reviews are capped to the first 100 per PR for speed. Does not measure incidents, reverts, feature adoption, mentorship/design work, or "
        "true review quality. Weights are adjustable to reflect leadership priorities."
    )


if __name__ == "__main__":
    main()
