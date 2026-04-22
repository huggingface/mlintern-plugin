# mlintern Claude Plugin

Use `ml-intern` from inside Claude Code.

[ML Intern](https://github.com/huggingface/ml-intern) is an open-source ML engineer agent from Hugging Face that can research, write, and run ML workflows such as fine-tuning, evaluation, and shipping models. This plugin makes it easy to invoke that workflow from Claude Code.

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

The `/plugin ...` .

Install `ml-intern` first (skip if already installed):

```bash
git clone git@github.com:huggingface/ml-intern.git
cd ml-intern
uv sync
uv tool install -e .
```

Then install this Claude plugin (the commands below are Claude Code slash commands and should be run inside an active Claude Code session, not your regular shell terminal):

```bash
/plugin marketplace add huggingface/mlintern-plugin
/plugin install mlintern@huggingface-mlintern
/reload-plugins
/mlintern:setup
```

## Run it!

Now, in your Claude Code session, you can do:

```bash
/mlintern --background "Find an Urdu speech dataset and fine-tune a small Whisper model in under 1 hour, evaluate WER, and share the model link and metrics.
/mlintern:status
/mlintern:result
```
