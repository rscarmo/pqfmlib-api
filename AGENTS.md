# Repository Guidelines

## Working Mode

For non-trivial changes, inspect the relevant code before modifying files.

Before implementation:
- understand the existing architecture and behavior;
- identify the files and components that will be affected;
- propose an implementation plan;
- split complex work into small, reviewable phases.

Present the plan to the user and wait for explicit approval before starting implementation.

After approval, implement only the approved phase.

Do not automatically continue to the next phase. Stop after each phase and wait for user approval.

Small, obvious, and local changes may be implemented directly when explicitly requested.

## Implementation Log

After each implementation phase, provide a concise operational log containing:

- what was changed;
- why it was changed;
- files modified;
- important implementation decisions;
- tests or commands executed;
- test results;
- remaining risks, problems, or open questions;
- the proposed next phase.

Do not provide unnecessary internal reasoning. Focus on decisions, actions, and verifiable results.

## Testing

Run the relevant tests after each implementation phase.

Add or update tests when behavior changes.

Prefer deterministic tests and avoid tests that depend on external credentials, network services, or real quantum hardware unless explicitly requested.

Do not consider a phase complete while relevant tests are failing without clearly reporting the failures.

## Safety and Scope

Preserve existing behavior and backward compatibility unless the task explicitly requires changing them.

Prefer the smallest change that correctly solves the problem.

Do not introduce new dependencies without explicit approval.

Do not submit jobs to real quantum hardware unless explicitly requested.

Do not modify unrelated code merely to clean up or refactor it.

## Git

Do not commit, push, create pull requests, or modify remote branches unless explicitly requested.

Do not run destructive Git commands or discard existing user changes without explicit approval.

Before reporting a phase as complete, inspect the resulting diff and make sure the changes are limited to the approved scope.

## Project Conventions

Follow the repository's existing structure, naming conventions, formatting, and testing patterns.

Prefer existing project tools and utilities over introducing new ones.