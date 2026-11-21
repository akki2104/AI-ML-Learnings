# Revision — E002: Tool Calling & Agent Loops

**Topic completed:** 2026-11-21 (passed after retest)
**Spaced repetition:** +1d (2026-11-22) · +3d (2026-11-24) · +7d (2026-11-28) · +15d (2026-12-06) · +30d (2026-12-21)

| Date | Reviewed? | Score |
|------|-----------|-------|
| 2026-11-22 | ⬜ | — |
| 2026-11-24 | ⬜ | — |
| 2026-11-28 | ⬜ | — |
| 2026-12-06 | ⬜ | — |
| 2026-12-21 | ⬜ | — |

**Known weak spots:** the two appended messages + whole-history resend; tools go in the `tools`
parameter (provider injects them), not your system prompt; one slow tool returns an error, only the
overall budget ends the loop; name the risk "indirect prompt injection".

## 30-second explanation
The model never executes anything. It returns a structured tool call (name, JSON-string arguments,
id). My code validates the arguments (errors go back as the tool result so the model self-corrects),
runs the tool with a timeout, appends the echoed assistant tool-call message plus a `tool` message
with the same `tool_call_id`, and resends the whole history. The loop ends when the model replies
without a tool call, or a round / time / token cap or repeat detector stops it.

## Active Recall Questions
1. Message by message: what the model returns for a tool call and what you send back.
2. How does the model decide its next call, and every way the loop ends?
3. Why validate when a schema exists? What happens on failure? What can strict mode never catch?
4. Parallel vs sequential tool calls — tradeoff, and when to turn parallel off.
5. Agent reads reviews/retailer pages: the risk, four defences, and when to use a pipeline instead.

## Answer Key
1. `finish_reason:"tool_calls"`, `tool_calls:[{id, name, arguments(JSON string)}]`. Append (a) the
   assistant message with its tool_calls, (b) a `tool` message `{tool_call_id, content}`; resend full
   history (model is stateless). Mismatched/missing ids → the API rejects the request.
2. Tools passed via `tools`; provider injects them into the prompt; model fine-tuned to emit calls;
   the description drives the choice. Ends: no tool call · round cap · overall time budget · token
   budget · repeated identical call. A single tool timeout returns `{"error":"timeout"}` instead.
3. Schema is a request; enforcement varies by provider. Validate (Pydantic), return the error as the
   tool result → self-correction. Strict mode guarantees shape, never meaning: SHIP-999 doesn't
   exist; SHIP-007 belongs to another customer.
4. Parallel: one round for independent lookups, but some models garble multi-calls. Sequential:
   slower, clean, natural when call 2 needs call 1's result. Turn parallel off for such models.
5. Indirect prompt injection (instructions hidden in data). Defences: least privilege (server-
   injected ids, brand-scoped, minimal tools); delimiters + "data, not instructions"; no publishing
   without human approval or a deterministic check; validate output (rule checker + judge vs product
   data). Pipeline when steps are fixed: brand data → retailer rules → generate → validate → publish
   after approval.
