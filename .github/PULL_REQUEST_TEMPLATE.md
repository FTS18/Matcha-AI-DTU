## Description

Provide a detailed summary of the changes in this pull request. Explain what problem this solves, what was changed, and why. Do not just repeat the commit message.

**Related Issue**: Closes # (required all PRs must be linked to an issue)

If this PR partially addresses an issue, write "Partially addresses #" and describe what remains to be done.

---

## Type of Change

Select all that apply:

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality described in the roadmap or an approved issue)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected requires major version bump discussion)
- [ ] Documentation improvement (changes to .md files, code comments, or docstrings only)
- [ ] Refactor (code restructuring without changing external behavior)
- [ ] Performance improvement (measurably faster or more efficient without changing behavior)
- [ ] Test addition (new unit or integration tests)
- [ ] Dependency update (changes to package.json, requirements.txt, or lock files)
- [ ] Infrastructure change (Docker, CI/CD, GitHub Actions, turbo.json)

---

## What Services Does This PR Touch?

- [ ] Frontend `apps/web/`
- [ ] Mobile App `apps/mobile/`
- [ ] Orchestrator `services/orchestrator/`
- [ ] Inference Engine `services/inference/`
- [ ] Shared Packages `packages/`
- [ ] Database Schema `packages/database/prisma/schema.prisma` (requires migration)
- [ ] Contracts / API Types `packages/contracts/`
- [ ] Infrastructure / Docker / CI

---

## How Has This Been Tested?

Describe the tests you ran to verify your changes. Provide enough detail that a maintainer can reproduce your testing.

**Manual Testing Checklist:**

- [ ] Tested locally with the full stack running (`docker-compose up -d` + `npx turbo run dev`).
- [ ] Uploaded a test video and verified the analysis pipeline completes without errors.
- [ ] Verified the frontend displays results correctly after analysis.
- [ ] Tested on at least one mobile viewport width (375px) if frontend changes were made.
- [ ] Verified no regressions in other tabs (Highlights, Events, Analytics) if changes were made to the match detail page.

**Inference Pipeline Checklist (if `services/inference/` was changed):**

- [ ] Ran a full analysis on a test video and confirmed all 5 phases completed.
- [ ] Confirmed the WebSocket progress callbacks were received correctly on the frontend.
- [ ] Verified log output in the Python terminal does not contain unexpected errors.

**Database Checklist (if `schema.prisma` was changed):**

- [ ] Created a migration with `npx prisma migrate dev --name <description>`.
- [ ] Confirmed the migration applied cleanly to a fresh database.
- [ ] Confirmed the Prisma client was regenerated (`npx prisma generate`).
- [ ] Updated the corresponding Zod schema in `packages/contracts/` if any new fields were added to API payloads.

---

## Checklist

- [ ] My code follows the style guidelines described in `docs/CONTRIBUTING.md`.
- [ ] I have performed a self-review of my code and removed all debug `console.log` and `print` statements.
- [ ] I have added comments to any code that is complex or non-obvious, especially in the Python inference pipeline.
- [ ] I have made corresponding changes to the documentation (README, SETUP.md, relevant docs/ files) if the behavior of the system has changed.
- [ ] I have updated the `ROADMAP.md` if this PR completes a roadmap item.
- [ ] My changes generate no new TypeScript type errors (`npx tsc --noEmit` passes).
- [ ] My changes generate no new ESLint warnings (`npm run lint` passes).
- [ ] The Turborepo build passes (`npx turbo run build`).
- [ ] I have NOT committed any `.env` files or API keys.
- [ ] If this PR adds new environment variables, I have added them to the relevant `.env.example` files and documented them in `docs/ARCHITECTURE.md`.

---

## Screenshots or Recordings (if applicable)

If your changes affect the UI, include before and after screenshots or a short screen recording. This is required for any frontend changes.

**Before:** (screenshot or description)

**After:** (screenshot or description)

---

## Breaking Changes

If this is a breaking change, describe exactly what will break for existing users and what they need to do to migrate.

Example: "This PR renames the `ttsAudioUrl` field on the Match API response to `highlightAudioUrl`. Any code that reads `match.ttsAudioUrl` must be updated."

---

## Additional Notes for Reviewers

Any specific areas you would like reviewers to focus on, or any decisions you made during implementation that you want feedback on.
