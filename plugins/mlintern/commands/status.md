---
description: Show ml-intern plugin job status
argument-hint: '[job-id] [--wait]'
disable-model-invocation: true
allowed-tools: Bash(node:*)
---

Run:

```bash
node "${CLAUDE_PLUGIN_ROOT}/scripts/mlintern-companion.mjs" status $ARGUMENTS
```

Return stdout exactly as produced.
