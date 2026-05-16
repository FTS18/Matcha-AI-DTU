# Your First Contribution to Matcha-AI-DTU

This guide is written specifically for people who have never contributed to an open-source project before. It explains every step — from finding the repository on GitHub all the way to seeing your name in the contributors list — in plain English. No prior Git experience is assumed.

If you already know how to fork, clone, branch, commit, and open a PR, you can skip to `./CONTRIBUTING.md` which has the project-specific technical details.

---

## Table of Contents

1. [What is Open Source Contribution?](#1-what-is-open-source-contribution)
2. [What Will You Actually Do?](#2-what-will-you-actually-do)
3. [Step 1 — Find Something to Work On](#step-1--find-something-to-work-on)
4. [Step 2 — Fork the Repository](#step-2--fork-the-repository)
5. [Step 3 — Clone Your Fork to Your Computer](#step-3--clone-your-fork-to-your-computer)
6. [Step 4 — Set Up the Project Locally](#step-4--set-up-the-project-locally)
7. [Step 5 — Create a Branch](#step-5--create-a-branch)
8. [Step 6 — Make Your Changes](#step-6--make-your-changes)
9. [Step 7 — Commit Your Changes](#step-7--commit-your-changes)
10. [Step 8 — Push to GitHub](#step-8--push-to-github)
11. [Step 9 — Open a Pull Request](#step-9--open-a-pull-request)
12. [Step 10 — The Review Process](#step-10--the-review-process)
13. [Common Mistakes and How to Avoid Them](#common-mistakes-and-how-to-avoid-them)
14. [Git Quick Reference](#git-quick-reference)

---

## 1. What is Open Source Contribution?

Open source means the source code of a project is publicly available for anyone to read, use, and improve. When you contribute to an open-source project, you are proposing a change (a bug fix, a new feature, improved documentation) that the project's maintainers can review and merge into the official codebase.

You do not have permission to directly edit the original repository. Instead, the contribution process works like this:

1. You make your own personal copy of the repository (called a "fork").
2. You make changes on your copy.
3. You submit a "pull request" asking the maintainers to pull your changes into the original.

This system means anyone can propose changes without risk of accidentally breaking the main project.

---

## 2. What Will You Actually Do?

A contribution can be any of these things:

- Fixing a typo or improving wording in a documentation file (the easiest possible contribution).
- Adding a missing section to `./SETUP.md` or `./FAQ.md`.
- Fixing a small bug in the frontend (a CSS issue, a broken link, a missing error message).
- Adding a new feature from the `./ROADMAP.md`.
- Adding tests.
- Improving error messages in the Python inference pipeline.

For your first contribution, pick something small. A 5-line documentation fix that genuinely helps future contributors is more valuable than a large, half-finished feature.

---

## Step 1 — Find Something to Work On

### Option A: Look at Open Issues

Go to the GitHub repository page and click the **Issues** tab. Look for issues labeled:

- `good first issue` — specifically chosen as suitable for newcomers.
- `documentation` — usually requires no coding, just clear writing.
- `beginner` — simple tasks that do not require deep knowledge of the codebase.

Read the issue description carefully. If it is unclear what needs to be done, ask for clarification by leaving a comment on the issue before starting work.

### Option B: Browse the Roadmap

Open `./ROADMAP.md` and look for items labeled `[Beginner]`. Find one that interests you. Check if a corresponding GitHub issue already exists. If not, open a new issue referencing the roadmap item and ask to be assigned.

### Claiming an Issue

Before starting work, leave a comment on the issue saying you would like to work on it. A maintainer will assign it to you. Do not start coding until you are assigned — otherwise two people might work on the same thing simultaneously.

---

## Step 2 — Fork the Repository

Forking creates your own copy of the project under your GitHub account.

1. Go to the Matcha-AI-DTU repository page on GitHub.
2. Click the **Fork** button in the top-right corner of the page.
3. GitHub will ask where to fork. Select your personal account.
4. After a few seconds, you will be taken to your copy of the repository at `https://github.com/YOUR-USERNAME/Matcha-AI-DTU`.

Your fork is independent. Changes you make there do not affect the original repository until you open a Pull Request.

---

## Step 3 — Clone Your Fork to Your Computer

Cloning downloads the repository files to your local machine so you can edit them.

Open a terminal (PowerShell on Windows, Terminal on macOS/Linux) and run:

```bash
git clone https://github.com/YOUR-USERNAME/Matcha-AI-DTU.git
```

Replace `YOUR-USERNAME` with your actual GitHub username.

This creates a folder called `Matcha-AI-DTU` in your current directory. Navigate into it:

```bash
cd Matcha-AI-DTU
```

Now connect your local copy to the original repository (called "upstream") so you can pull future updates:

```bash
git remote add upstream https://github.com/FTs18/Matcha-AI-DTU.git
```

Verify both remotes exist:

```bash
git remote -v
```

You should see:

```
origin

https://github.com/YOUR-USERNAME/Matcha-AI-DTU.git (fetch)
origin

https://github.com/YOUR-USERNAME/Matcha-AI-DTU.git (push)
upstream https://github.com/FTs18/Matcha-AI-DTU.git (fetch)
upstream https://github.com/FTs18/Matcha-AI-DTU.git (push)
```

---

## Step 4 — Set Up the Project Locally

Follow `./SETUP.md` to get the full project running on your machine. For a documentation-only contribution, you do not need to run the servers — you only need a text editor. For a code contribution, you will need the full stack running.

If you run into setup problems, check `./FAQ.md` first. If your issue is not there, ask in GitHub Discussions.

---

## Step 5 — Create a Branch

A branch is an isolated workspace for your changes. You should never make changes directly on `main` or `dev`. Always create a new branch.

First, make sure your local `dev` branch is up to date with the original:

```bash
git checkout dev
git pull upstream dev
```

Then create your branch from `dev`:

```bash
git checkout -b your-branch-name
```

Choose a branch name that describes what you are doing. Follow these conventions:

| What you are doing      | Branch name format                       | Example                              |
| ----------------------- | ---------------------------------------- | ------------------------------------ |
| Adding a new feature    | `feature/issue-number-short-description` | `feature/42-add-dark-mode`           |
| Fixing a bug            | `bugfix/issue-number-short-description`  | `bugfix/17-fix-upload-error-message` |
| Improving documentation | `docs/short-description`                 | `docs/add-windows-venv-note`         |

You are now on your new branch. All changes you make will only exist on this branch until you merge them.

---

## Step 6 — Make Your Changes

Open the project in your code editor (VS Code is recommended). Make the changes needed to address the issue you claimed.

**Some tips:**

- Make one focused change at a time. If you notice an unrelated bug while working, open a separate issue for it rather than fixing it in the same PR.
- If you are editing a Markdown file (`.md`), check how it looks formatted by using VS Code's built-in Markdown preview (Ctrl+Shift+V).
- If you are editing code, make sure to test your changes by running the relevant service.
- Do not change unrelated files. Keep your changes minimal and focused.

---

## Step 7 — Commit Your Changes

A commit is a saved snapshot of your changes with a message describing what you did. Think of commits as checkpoints in your work.

First, check what files you have changed:

```bash
git status
```

Add the files you want to include in your commit:

```bash
# Add a specific file:
git add ./FAQ.md

# Add all changed files:
git add .
```

Write your commit message. This project uses the Conventional Commits format:

```bash
git commit -m "type: short description of what you did"
```

The `type` must be one of:

- `feat` — you added something new
- `fix` — you fixed a bug
- `docs` — you only changed documentation files
- `style` — formatting changes (spaces, commas) with no logic change
- `refactor` — code restructuring with no behavior change
- `chore` — updating config files, dependencies

Examples of good commit messages:

```
docs: add Windows venv activation note to ./SETUP.md
fix: resolve CORS error when Next.js starts on port 3001
feat: add skeleton loading states to match dashboard
docs: add GPU requirement clarification to ./FAQ.md
```

Examples of bad commit messages:

```
update
fix stuff
changes
```

You can make multiple commits as you work. They will all be included in your Pull Request.

---

## Step 8 — Push to GitHub

Pushing uploads your local commits to your fork on GitHub:

```bash
git push origin your-branch-name
```

The first time you push a new branch, Git will print a URL you can click to open a Pull Request — this is the fastest way to get to Step 9.

---

## Step 9 — Open a Pull Request

1. Go to your fork on GitHub (`https://github.com/YOUR-USERNAME/Matcha-AI-DTU`).
2. GitHub will usually show a yellow banner saying "your-branch-name had recent pushes" with a "Compare and pull request" button. Click it.
3. If the banner is not there, click the **Pull requests** tab and then click **New pull request**.
4. Make sure the base repository is the original Matcha-AI-DTU and the base branch is **`dev`** (not `main`). Make sure the head is your fork and your branch.
5. Fill in the Pull Request template that appears. Answer every section honestly. A complete PR description gets reviewed much faster than an empty one.
6. In the description, link to the issue you are solving by writing `Closes #42` (replace 42 with the actual issue number). This automatically closes the issue when the PR is merged.
7. Click **Create Pull Request**.

---

## Step 10 — The Review Process

After opening a PR, a maintainer will review your code. This is a normal and collaborative process — do not be discouraged by review comments. Every professional developer goes through code review.

**What happens during review:**

- The reviewer may leave comments on specific lines of code asking questions or requesting changes.
- You can respond to comments by replying in the PR thread.
- If changes are requested, make them locally, commit, and push to the same branch. The PR automatically updates.
- Once the reviewer is satisfied, they will approve the PR and merge it.

**What to expect:**

- Reviews typically happen within 48-72 hours. If you do not hear back in 3 days, leave a friendly comment to ping the reviewers.
- It is normal for a PR to go through 2-3 rounds of review before being merged.
- A "request for changes" is not a rejection. It means the reviewer wants to help you improve the contribution.

**After your PR is merged:**

- You will appear in the project's contributor list.
- The issue is automatically closed.
- Your changes are live in the `dev` branch and will reach `main` on the next release.
- You can delete your local branch: `git branch -d your-branch-name`

---

## Common Mistakes and How to Avoid Them

**Opening a PR against `main` instead of `dev`** Always check that the "base" branch in your PR is `dev`. The project's development happens on `dev`; `main` is only for stable releases.

**Not claiming the issue before starting** Two people end up working on the same thing. Always comment on the issue and wait to be assigned before writing any code.

**Making changes on `main` or `dev` directly** Always create a new branch with `git checkout -b branch-name`. Changes made directly on `dev` will cause conflicts when you try to sync with upstream.

**Committing `.env` files** Never commit `.env` files. They contain API keys and passwords. Check `git status` before committing and make sure no `.env` files appear. They should be listed in `.gitignore` already, but double-check.

**A very large PR with many unrelated changes** Reviewers struggle with PRs that touch dozens of files across unrelated areas. Keep each PR focused on one issue. If you find other things to fix, open separate issues and PRs.

**Not syncing with upstream before starting work** If you start from an outdated `dev` branch, your PR may conflict with changes that were merged since you last synced. Always run `git pull upstream dev` before creating a new branch.

---

## Git Quick Reference

```bash
# Download the repository to your computer
git clone https://github.com/YOUR-USERNAME/Matcha-AI-DTU.git

# Add a connection to the original repo
git remote add upstream https://github.com/FTs18/Matcha-AI-DTU.git

# Get the latest changes from the original repo
git fetch upstream
git checkout dev
git merge upstream/dev

# Create a new branch and switch to it
git checkout -b feature/42-my-feature

# See what files you have changed
git status

# See the actual changes you made
git diff

# Stage a specific file for committing
git add path/to/file.md

# Stage all changed files
git add .

# Save your changes with a message
git commit -m "docs: improve setup instructions for Windows"

# Upload your branch to GitHub
git push origin feature/42-my-feature

# Switch to a different branch
git checkout dev

# See the list of all your branches
git branch

# Delete a branch you no longer need (after your PR is merged)
git branch -d feature/42-my-feature
```

---

You are ready. Go find an issue, claim it, and make your first contribution. The maintainers are here to help — do not hesitate to ask questions in GitHub Discussions.
