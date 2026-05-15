# GitHub Discussions Setup Guide

This file documents how GitHub Discussions is configured for Matcha-AI-DTU and what each category is for. It is intended for maintainers setting up a new fork of this project.

---

## Enabling Discussions

1. Go to the repository on GitHub.
2. Click **Settings** (the gear icon in the top navigation).
3. Scroll down to the **Features** section.
4. Check the **Discussions** checkbox.
5. Click **Save**.

Discussions is now enabled. A **Discussions** tab will appear in the repository navigation.

---

## Recommended Category Setup

After enabling Discussions, go to the **Discussions** tab and click the pencil icon next to "Categories" to configure them.

Create the following categories:

| Category Name | Format | Purpose |
| --- | --- | --- |
| Announcements | Announcement | Maintainer-only posts about new releases, GSSOC updates, and project news |
| Introduce Yourself | Open Ended | New contributors introduce themselves and say what they want to work on |
| Questions and Help | Question and Answer | Any questions about setup, the codebase, or how things work |
| Ideas Board | Open Ended | Suggest new features or discuss improvements before opening a formal issue |
| Show and Tell | Open Ended | Share what you have built or contributed to the project |
| GSSOC 2026 | Open Ended | GSSOC-specific discussions, mentor-mentee communication, and progress updates |

---

## Starter Threads to Create

Once categories are configured, create the following pinned starter threads to make new contributors feel welcome:

### Thread 1 Introduce Yourself

**Category**: Introduce Yourself **Title**: Introduce yourself tell us what you want to work on **Body**:

```
Welcome to Matcha-AI-DTU! We are glad you are here.

To get started, drop a comment below with:
- Your name (or GitHub handle)
- What brought you to this project (GSSOC, personal interest, a specific feature?)
- What area you are most interested in working on (frontend, backend, Python AI pipeline, documentation)
- Your experience level with the relevant technologies (complete beginner / some experience / experienced)

If you are brand new to open source, do not worry  this project has plenty of beginner-friendly tasks.
Read FIRST_CONTRIBUTION.md for a step-by-step guide to your first PR.

Maintainers check this thread regularly and will point you to a good first issue based on your interests.
```

### Thread 2 Questions and Help

**Category**: Questions and Help **Title**: Questions about the codebase ask anything here **Body**:

```
This is the right place to ask any question about the project, no matter how simple.

Before asking, please check:
- docs/FAQ.md  answers to the most common questions
- GLOSSARY.md  plain-English definitions of technical terms
- docs/TROUBLESHOOTING.md  known errors and their fixes
- SETUP.md  the complete setup guide

If your question is not answered in those documents, post it here with:
- What you were trying to do
- What you expected to happen
- What actually happened (paste any error messages in full)
- What you have already tried

There are no stupid questions. If you are confused about something, other contributors probably are too.
```

### Thread 3 Ideas Board

**Category**: Ideas Board **Title**: Ideas board suggest features before opening an issue **Body**:

```
Have an idea for a new feature or improvement that is not already in ROADMAP.md?

Post it here first for community feedback before opening a formal Feature Request issue.
This prevents opening issues for things that are out of scope or already planned under a different name.

A good idea post includes:
- What problem does this solve? Who benefits?
- Rough idea of how it might be implemented (even a one-sentence sketch is helpful)
- Are you interested in implementing it yourself?

If maintainers agree the idea is a good fit, we will convert it into a GitHub issue and add it to the roadmap.
```
