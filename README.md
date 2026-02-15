📊 PostHog Engineering Impact Dashboard (PR-Only MVP)
A lightweight, rate-limit-aware Streamlit dashboard that analyzes GitHub PR metadata to surface multi-dimensional engineering impact signals.
⚠️ This is a decision-support tool, not a performance ranking system.

🎯 Goal
Given a strict time constraint, the goal was to:
* Define a defensible model of engineering impact
* Use only observable GitHub PR metadata
* Keep the system bounded and rate-limit aware
* Build a clean, interactive dashboard for engineering leadership
This MVP focuses exclusively on PR-level signals and intentionally excludes incident linkage, NLP classification, and identity reconciliation.

🧠 How “Impact” Is Defined
Engineering impact is modeled as a weighted composite of observable PR signals:
1️⃣ Delivery (log-scaled output)
* Sum of log(1 + additions + deletions)
* Prevents large PRs from dominating
* Rewards meaningful shipped work
2️⃣ Reviews Written (team multiplier)
* Count of distinct PRs reviewed
* Deduplicated per (PR, reviewer)
* Bot reviewers filtered
3️⃣ Cycle Time (execution efficiency)
* Median time from PR creation → merge
* Capped at 14 days to reduce outlier distortion
* Inverted so faster merges score higher
4️⃣ Leverage (optional)
* Sum of log(1 + changedFiles)
* Proxy for cross-cutting/system-wide changes
Each metric is min-max normalized across engineers and combined via adjustable weights.

⚖️ Composite Score

Impact Score =
  w_delivery * delivery_n
+ w_reviews  * reviews_n
+ w_cycle    * cycle_n
+ w_leverage * leverage_n
Weights are configurable in the UI and auto-normalized.

🔒 Scope Constraints (Intentional)
This MVP explicitly excludes:
* ❌ Incident or revert linkage
* ❌ Bug attribution
* ❌ NLP classification of PR titles
* ❌ Commit-level analysis
* ❌ Author alias reconciliation
* ❌ Deep historical trend modeling
Why?
Given the time constraint, the focus was on:
* Structural GitHub signals
* Reproducibility
* Simplicity
* Interpretability
* Reliability under API rate limits

🛠 Architecture Overview
Data Sources
* GitHub Search API → merged PR numbers
* GitHub GraphQL API → PR metadata + reviews
* REST fallback → limited mode without token
Rate Limit Strategy
* Token-aware fetch logic
* Dynamic batch sizing based on GraphQL remaining quota
* Hard cap on PR volume
* REST-only limited mode if no token provided

🚀 Running the App
1️⃣ Install dependencies

pip install -r requirements.txt
2️⃣ Run locally

streamlit run app.py
3️⃣ Recommended: Provide a GitHub token
Paste a fine-grained token in the sidebar (read access to public repos).
Without a token:
* Limited mode activates
* Up to 50 PRs
* Cycle-time-only scoring

📦 requirements.txt

streamlit>=1.31
pandas>=2.0
requests>=2.31

📊 Dashboard Features
* Adjustable weight sliders
* Top 5 engineers by composite score
* Single breakdown visualization
* PR drilldown per engineer
* Built-in rate limit diagnostics
* REST fallback for limited mode

🧩 Design Decisions
* Log scaling prevents PR size gaming.
* Median merge time reduces outlier distortion.
* Review deduplication avoids counting multiple review states.
* Bot filtering improves signal quality.
* Activity floor logic reduces noise from extremely low-volume contributors.
* PR-only scope maximizes reliability within time constraints.

⚠️ Limitations
This dashboard captures observable GitHub PR signals only.
It does not measure:
* Mentorship
* Design leadership
* Incident response
* Feature adoption
* Business impact
* Long-term architectural improvements
Scores are heuristic and intended to:
Surface signals and start conversations, not replace human judgment.

🔮 Future Iterations
If extended beyond MVP:
* Incorporate revert/incident linkage
* Normalize per active month (tenure bias reduction)
* Add subsystem weighting
* Add distribution visualization (avoid leaderboard framing)
* Validate metrics against downstream outcomes
* Add confidence intervals for stability analysis

🏁 Final Note
This MVP prioritizes:
* Pragmatism over exhaustiveness
* Interpretability over statistical sophistication
* Reliability under API constraints
* Multi-dimensional signals over single-metric ranking
It is designed to help engineering leaders quickly identify meaningful patterns in recent PR activity — while clearly communicating its limitations.
