# CLAUDE.md — Instructions for Claude Code

This file is read automatically at the start of every session in this repository.
Use it to set preferences, conventions, and constraints.

---

## Language & Style

- This is a Julia project. All source code is Julia unless otherwise noted.
- Follow existing code style in each file rather than imposing a uniform style
  across the whole codebase.
- Do not add docstrings, comments, or type annotations to code you did not change.
- Do not refactor surrounding code when fixing a bug or adding a feature — keep
  changes focused.
- Do not add or remove any whitespaces except in the lines you are editing already.
- Try to keep lines under 80 characters, with 92 characters the absolute maximum.

<!-- Adjust the line length limit if the project has a preference: -->

---

## Workflow

- Do not commit changes unless explicitly asked to.
- Do not push to remote unless explicitly asked to.
- Before editing a file, read it first.
- Prefer editing existing files over creating new ones.

<!-- Uncomment and adjust if you want a specific test command run after changes: -->
<!-- ## Testing -->
<!-- - After making changes, run: `julia --project=. -e 'using Pkg; Pkg.test()'` -->
<!-- - All tests must pass before considering a task complete. -->

---

## Julia-specific

- Prefer in-place (`!`) variants of functions when performance matters.
- Do not introduce type instabilities. If a change may affect type inference,
  note it explicitly.
- Do not add dependencies to `Project.toml` without asking first.
- GPU kernels use KernelAbstractions — do not introduce CUDA.jl-specific code
  in shared paths.
- It is ok to use features from the latest Julia versions. If you do, provide brief summaries and comments.

<!-- Uncomment if you want a specific Julia version enforced: -->
<!-- - Target Julia 1.11+. Do not use features from 1.12 or later. -->

---

## Commit style

<!-- Adjust to match the project's existing commit history style. -->
<!-- Common options: Conventional Commits, plain imperative, etc. -->

- Use short imperative commit messages (e.g. "Remove dead simpletrans.jl").
- Do not add "Co-Authored-By" lines unless asked.

<!-- Uncomment to enforce Conventional Commits: -->
<!-- - Use Conventional Commits format: `feat:`, `fix:`, `refactor:`, `docs:`, etc. -->

---

## What NOT to do

- Do not silently change behaviour — if a refactor changes observable output,
  flag it before proceeding.
- Do not delete files without confirmation, even if they appear unused.
- Do not open pull requests without being asked.

<!-- Add any other hard constraints here, e.g.: -->
<!-- - Never modify Project.toml or Manifest.toml. -->
<!-- - Never touch the ext/ directory. -->
