# Pushing this repo to GitHub

This project already has a local git repo initialized (with one commit).
I can't push to your GitHub account directly from here (no access to your
credentials, and it shouldn't have any), so do this yourself — it's 3 steps:

## 1. Create an empty repo on GitHub
Go to https://github.com/new
- Repository name: `vdocrag-project` (or whatever you like)
- **Do NOT** initialize with a README/.gitignore/license (this repo already has them —
  ticking those boxes would create conflicting files)
- Click "Create repository"

## 2. Copy the remote URL
GitHub will show you a URL like:
```
https://github.com/<your-username>/vdocrag-project.git
```

## 3. Push
From inside the `vdocrag-project` folder on your machine:
```bash
git remote add origin https://github.com/<your-username>/vdocrag-project.git
git branch -M main
git push -u origin main
```

If prompted for credentials, use a GitHub Personal Access Token (not your
password) — GitHub will show you how to create one if you don't have one,
or use `gh auth login` if you have the GitHub CLI installed.

## Working with your friend's RTX GPU going forward
Simplest flow:
1. You develop/commit on the MacBook, `git push`.
2. Your friend `git pull`s on the RTX machine, runs `bash scripts/train_retriever_lora.sh`
   (etc.), and once done, commits just the small config/log files back
   (the actual model weights + embeddings are already gitignored — don't
   commit multi-GB checkpoint files to GitHub; upload those to Google Drive/
   Hugging Face Hub instead and just link them in the README).
