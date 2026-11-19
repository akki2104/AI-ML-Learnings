# Session Handoff — written 2026-11-19 (cloud session → local session)

Read this once at the start of the next session, then follow `CLAUDE.md` as usual.

## Where we are
- Done: A001, A002, B001 (3 / 36). Next: **D001 When to Fine-tune vs RAG vs Prompting**.
- Mode: **interview fast-track** — the learner has an interview soon (date and company type NOT yet given; ask once at the start).
  Order: D001 → B007 → B004 → B006 → B009 → E004 → E003 → E002 → F001 (mock interview).
  Tier 2 if time: B002, B005, C003+C001, E005. Skipped for now: A003, A004, C004, D003–D005, E001, E006, F005.
- Revisions: A002 (+1d due 2026-11-19), B001 (+1d due 2026-11-20); A001 revisions overdue. Surface them, never block on them.

## How the learner wants lessons (also in `AI_ML_MASTER_GUIDE.md §3.1`)
- Copy the System Design lesson voice (`../System-Design-Learnings/Topics/042_Consistent_Hashing.md`).
- Open with "Why This Topic Exists", then "The Problem, Precisely" with a concrete example BEFORE the solution.
- Define every jargon term at first use: plain meaning + why it exists. He said earlier lessons felt jargon-heavy and irritating.
- Always state the tradeoff; end with Tech Decision Box, interview sentence, mistakes table, interview questions, summary.
- Deliver the full lesson in one pass, ask all recall questions together at the end, then score them.
  Topic is marked complete only after a passed retest of anything he missed (A001/A002/B001 all went this way).

## Working rules the learner set
- Push straight to `main` in this repo (no PRs, no feature branch).
- Commit as `akki2104 <ay0832760@gmail.com>`.
- Log each wrong/incomplete answer in `InterviewMistakes.md`; update Progress, Revision file, CLAUDE.md "Curriculum Position".

## Recurring weak spots to drill
- Always state the tradeoff of a mechanism (he often omits it).
- First debugging question for RAG: "was the right chunk in the prompt?"
- Fine-tuning is poor at injecting facts / no citations.
- Take identity (e.g. team id) from the auth token on the server, never from the client.
- Normalized vectors → dot = cosine = same ranking as L2; new embedding model = re-embed everything.
