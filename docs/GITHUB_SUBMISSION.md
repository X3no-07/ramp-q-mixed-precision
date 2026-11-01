# Working with the GitHub repository

Target repository: [X3no-07/ramp-q-mixed-precision](https://github.com/X3no-07/ramp-q-mixed-precision).

Run the packaging check before a source update:

```bash
python tools/check_repository.py
git status
git diff --cached --stat
```

When updating an existing clone, fetch the current remote state, preserve its history, and commit the reviewed changes on top of it. Push normally; do not rewrite the existing main-branch history to replace a snapshot.

```bash
git add --all
git commit -m "Update research source and reports"
git push origin main
```

Use your own configured Git identity and authentication. Keep local settings, credentials, datasets, and large results out of the staged tree. Refer to [artifact availability](DATA_AND_ARTIFACTS.md) for historical releases and excluded evidence.
