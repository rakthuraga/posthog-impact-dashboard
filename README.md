Engineering Impact Dashboard
PostHog Repository Analysis (PR-Only MVP)
Overview
This project analyzes GitHub pull request activity from the public PostHog repository and produces an interactive dashboard identifying engineers with strong “impact signals.”
The intended audience is a busy engineering leader who:
* Understands the team context
* Does not have time to read every PR
* Wants high-signal conversation starters
⚠️ This is not a performance ranking tool.It is designed to surface patterns for discussion.

What Does “Impact” Mean?
In this MVP, impact is approximated using observable pull request metadata:
1. Delivery – Volume of meaningful code changes (log-scaled lines changed)
2. Reviews Written – Contribution to others’ work (collaboration signal)
3. Cycle Time – Speed from PR creation to merge
4. Leverage – Breadth of change (log-scaled files touched)
These are intentionally simple, auditable metrics that:
* Are hard to game
* Are explainable
* Avoid heavy inference or subjective modeling

Scoring Model
Each engineer receives a normalized composite score:

Impact Score =
  w_delivery * delivery_n
+ w_reviews  * reviews_n
+ w_cycle    * cycle_n
+ w_leverage * leverage_n
Where:
* Metrics are min-max normalized (0–1 range)
* Weights are user-adjustable in the UI
* Scores are meant for relative comparison within the selected window
Improvements for Robustness
To prevent misleading rankings:
* Engineers with fewer than 3 merged PRs are excluded
* A consistency multiplier rewards sustained contribution:

impact_score *= (prs_merged / max_prs)
* PR size is log-scaled to prevent large single PR dominance
* Merge time is capped at 14 days to avoid long-tail distortion

Limited Mode vs Full Mode
Limited Mode (No Token)
* Uses REST-only endpoints
* Fetches up to 50 PRs
* Computes Cycle Time Snapshot only
* Delivery/review/leverage metrics are removed from UI
This avoids GitHub anonymous rate limits and keeps UX clear.

Full Mode (With GitHub Token)
* Uses GraphQL batching
* Fetches up to 400 merged PRs
* Enables:
    * Delivery
    * Reviews written
    * Leverage
    * Adjustable weights
* Rate-limit aware and bounded for safety

Why PR-Only?
This MVP intentionally excludes:
* Incident / revert linkage
* Production bug attribution
* Feature adoption metrics
* NLP analysis of PR descriptions
* Design / mentorship contributions
These require:
* Additional GitHub endpoints
* Cross-referencing issues
* Semantic inference
Given the assignment scope, the goal was to build:
* A robust, explainable baseline
* Something extensible
* Something that works reliably under rate limits

Key Design Decisions
1️⃣ Log Scaling
Large PRs can dominate raw metrics.Using log1p(lines_changed) dampens size outliers.

2️⃣ Median Merge Time
Median is more robust than mean for skewed distributions.

3️⃣ Normalization on Active Engineers Only
Prevents low-activity contributors from distorting score ranges.

4️⃣ Soft Consistency Reward
Prevents one-PR dominance while keeping model simple.

5️⃣ Rate Limit Awareness
* Search API capped
* GraphQL batched
* Stops early if remaining < 200
* Limited mode fallback
This ensures the dashboard remains stable and production-safe.

Running Locally
1. Install dependencies

pip install streamlit pandas requests
2. Run the app

streamlit run app.py
3. (Optional) Add GitHub token
Create a fine-grained token with read access to public repos.
Either:
* Paste in sidebar
* Or set environment variable:

export GITHUB_TOKEN=your_token_here

How to Use
1. Select lookback window
2. Select max PR cap
3. Adjust metric weights
4. Review:
    * Top 5 impact signals
    * Metric breakdown
    * PR drilldown
The output is meant to drive questions like:
* Why is this engineer’s cycle time unusually low?
* Is review contribution concentrated among a few people?
* Are large changes distributed or centralized?

Limitations
* Does not measure true code quality
* Does not measure customer impact
* Does not measure mentorship
* Does not measure design influence
* Reviews are capped at 100 per PR
* No identity reconciliation for aliases
This is a metadata-based approximation.

Future Extensions
If extended further, I would add:
* Revert rate detection
* Bug / issue linkage after merge
* Review depth (iterations per PR)
* Comment quality analysis
* AI assistance measurement (PR diff entropy)
* Ownership clustering by file path
* Cross-team collaboration graph

Takeaway
This dashboard is intentionally:
* Simple
* Transparent
* Explainable
* Adjustable
* Rate-limit safe
It surfaces signals — not verdicts.
The goal is to augment engineering leadership judgment, not replace it.
