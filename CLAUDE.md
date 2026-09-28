# CLAUDE.md — AI/ML Interview Prep Track

This file provides session guidance for this AI/ML learning repository.
This is a **separate track** from the System Design curriculum at `../system design/`.

## What This Repository Is

AI/ML interview preparation for Akash Yadav — LLM/AI Engineer roles at product companies.
Hands-on experience with RAG pipelines, vector search, cross-encoder reranking, and LLM agent orchestration.
Gap: interview framing — architectural decisions, tradeoffs, evaluation, "design an AI system."

## Session Workflow

**Every session, in this order:**

0. Verify the real current date from system context (never recall from memory — date errors
   corrupt spaced-repetition schedules).
1. Read `AI_ML_MASTER_GUIDE.md` — the constitution. This is the operating manual.
2. Read `AI_ML_Progress.md` — find current position and next topic.
3. Read `AI_ML_Schedule.md` — find today's phase and session target.
4. Read `AI_ML_TopicPriority.md` — show the Topic Briefing Card for today's topic.
5. Deliver the lesson per the 24-step structure in `AI_ML_MASTER_GUIDE.md §3`.
6. Update all tracking files after the session.
7. Commit and push.

## Daily Briefing Format

```
📅 Today: [date]
📍 AI/ML schedule: [phase + session target]
✅ Actually done: [list from AI_ML_Progress.md]
⚡ Pace: [ahead / on track / behind]
🔁 Revisions due: [list or "none"]
🎯 Today's goal: [specific topics]
```

## File Roles

| File | Role |
|------|------|
| `AI_ML_MASTER_GUIDE.md` | Constitution — read every session |
| `AI_ML_Progress.md` | Live tracker — update after every session |
| `AI_ML_TopicPriority.md` | Tier table — read before every topic |
| `AI_ML_TechChoices.md` | Decision playbook — append after each lesson |
| `AI_ML_Schedule.md` | Phase/session schedule — read each session |
| `Topics/AI_NNN_*.md` | Lesson files — one per topic |
| `Revision/Revision_AI_NNN.md` | Active recall files — one per completed topic |
| `CaseStudies/CS_FNN_*.md` | Case study lesson files |

## Commit Rule

After every completed topic:

```bash
git add .
git commit -m "Topic [ID]: [Topic Name] — lesson + revision + tracking files updated

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
git push origin main
```

## Teaching Rules

1. Explain first, quiz second — never Socratic-quiz on unintroduced material.
2. Full explanation uninterrupted — ask all questions together at the end.
3. Tier shapes session depth, not file length — always write full lesson files.
4. Revision is the learner's cadence — surface backlog but don't block new content.

## Track Integration

This track is independent of `../system design/` but shares the learner's time.
AI/ML case studies (Module F) use the same 7-step interview framework from the
system design track — leverage Akash's existing knowledge there.

## Curriculum Position (as of 2026-09-19)

- Topics completed: 0 / 36
- Current module: Not started
- Next topic: A001 Transformer Architecture & Attention
- Target: Late October 2026
