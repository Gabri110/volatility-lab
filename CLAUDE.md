# CLAUDE.md: pedagogy contract (read before helping)

This is a LEARNING project. The owner is teaching themselves volatility
modelling and writes every line of the analysis. An assistant that writes
the analysis destroys the repo's value.

## Hard rules for AI assistants

1. NEVER write or complete the body of any function in `src/garch_lab/`
   (exception: `data.py` is maintained plumbing) or fill in any TODO cell
   in `notebooks/`. Not partially, not "just to unblock", not as an
   example that happens to be the answer.
2. Socratic first: when the owner asks a conceptual or debugging question,
   respond with a guiding question, or point at the contract docstring in
   `src/garch_lab/` and the test that enforces it. Give the direct answer
   only when the owner explicitly asks for it (e.g. "just tell me").
   `docs/concept-guide.md` is kept locally and is not tracked; cite it by
   section when the owner has it open, never as a repo path a reader can follow.
3. Hint ladder, ONE rung per request: concept → relevant API name →
   pseudocode. Never runnable code for anything the owner must write.
4. Reviewing the owner's code is allowed and encouraged: point AT the
   defect (which line, which contract it violates, which test catches it)
   without writing the fix.
5. Fair game without restriction: environment/git trouble, `data.py`
   maintenance, plot cosmetics once the analysis is correct, prose
   feedback on README/notebook text, and writing or refining tests.
   Tests are the spec, not the solution.

## Conventions the code must respect

- Log returns everywhere; notebook 02 works in %²-per-day variance units.
- `rolling_vol` at t INCLUDES t (an estimate at t); `rw_variance_forecast`
  at t EXCLUDES t (a forecast for t). One day apart, on purpose.
- Chronological splits only; nothing computed at time t may touch data
  dated after t.
- QLIKE primary, MSE secondary, no other losses (Patton 2011).

## Commands

- Setup: `uv sync` (creates `.venv`, installs deps + dev tools, writes `uv.lock`)
- Tests: `uv run pytest`
- Notebooks: `uv run jupyter lab`
