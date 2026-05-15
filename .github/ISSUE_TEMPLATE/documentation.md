---
name: Documentation Improvement
about: Report missing, incorrect, or unclear documentation
title: "[DOCS] "
labels: ["documentation", "good first issue"]
assignees: []
---

## Which Document Needs Improvement?

Specify the file path or URL of the documentation that needs to be improved.

Examples:

- `README.md`
- `SETUP.md`
- `docs/ARCHITECTURE.md`
- `docs/API_REFERENCE.md`
- `docs/CONTRIBUTING.md`
- `ROADMAP.md`
- Code-level docstrings in `services/inference/app/core/analysis.py`
- Inline code comments in `apps/web/app/matches/[id]/page.tsx`

---

## What Is the Problem?

Select the type of documentation issue:

- [ ] Missing documentation something important is not documented at all
- [ ] Incorrect documentation something is documented but the information is wrong or outdated
- [ ] Unclear documentation the documentation exists but is confusing or ambiguous
- [ ] Incomplete documentation the documentation exists but is missing key details
- [ ] Broken links or formatting issues
- [ ] Missing code examples or tutorials
- [ ] Translation or language clarity issue

---

## Describe the Problem in Detail

Be specific. If a section is confusing, quote the specific text that confused you. If something is missing, describe exactly what information you expected to find and where you expected to find it.

Example: "The SETUP.md file says to run `python3 -m venv venv` on Windows, but on Windows the command is `python -m venv venv`. The `python3` command does not work in standard Windows Command Prompt or PowerShell unless Python was installed with a specific option."

---

## What Should the Documentation Say?

Describe the correction or addition you believe should be made. If possible, write the corrected text directly. Proposing a specific fix greatly speeds up the time to merge.

---

## Who Does This Affect?

- [ ] Newcomers and first-time contributors setting up the project
- [ ] Windows users
- [ ] macOS users
- [ ] Linux users
- [ ] Contributors working on the Python inference pipeline
- [ ] Contributors working on the NestJS orchestrator
- [ ] Contributors working on the Next.js frontend
- [ ] All contributors

---

## Have You Verified the Current State?

- [ ] I have read the current version of the document and confirmed the problem exists in the latest version of the `dev` branch.

---

## Are You Willing to Submit a PR for This?

Documentation improvements are an excellent first contribution and are highly valued by the maintainers.

- [ ] Yes, I would like to be assigned this issue and submit a PR with the fix.
- [ ] No, I am reporting the issue for someone else to fix.

---

## Additional Context

Add any screenshots, error messages, or terminal output that illustrates the documentation problem.
