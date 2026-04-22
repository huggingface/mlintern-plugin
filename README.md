# mlintern Claude Plugin

Use `ml-intern` from inside Claude Code.

## Experimental

This plugin is experimental and may change frequently. Expect rough edges.

## What this plugin provides

- `/mlintern "fine-tune a model"` to run an ML Intern task
- `/mlintern --background "..."` to run in background
- `/mlintern:status` to check running/recent jobs
- `/mlintern:result [job-id]` to fetch final output
- `/mlintern:cancel [job-id]` to cancel a running job
- `/mlintern:setup` to verify local setup

## Requirements

- Node.js 18+ (if you are already using Claude Code, you should already have this)
- `ml-intern` installed and available on PATH
- Auth configured for `ml-intern` (HF token and any model provider keys you use)

The plugin install step itself does not currently install `ml-intern` for you. That keeps plugin install predictable across environments, but means `ml-intern` still needs to be installed separately on your machine.

## Install in Claude Code

```bash
/plugin marketplace add huggingface/mlintern-plugin
/plugin install mlintern@huggingface-mlintern
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
