# mlintern Claude Plugin

Use `ml-intern` from inside Claude Code.

## What this plugin provides

- `/mlintern "fine-tune a model"` to run an ML Intern task
- `/mlintern --background "..."` to run in background
- `/mlintern:status` to check running/recent jobs
- `/mlintern:result [job-id]` to fetch final output
- `/mlintern:cancel [job-id]` to cancel a running job
- `/mlintern:setup` to verify local setup

## Requirements

- Node.js 18+
- `ml-intern` installed and available on PATH
- Auth configured for `ml-intern` (HF token and any model provider keys you use)

## Install in Claude Code

```bash
/plugin marketplace add abidlabs/mlintern-plugin
/plugin install mlintern@abidlabs-mlintern
/reload-plugins
/mlintern:setup
```

## Example

```bash
/mlintern "fine-tune llama on my dataset"
/mlintern --background "train a reward model and report metrics"
/mlintern:status
/mlintern:result
```
