---
description: Show final output for a finished ml-intern job
argument-hint: '[job-id]'
disable-model-invocation: true
allowed-tools: Bash(node:*)
---

Run:

```bash
node "${CLAUDE_PLUGIN_ROOT}/scripts/mlintern-companion.mjs" result $ARGUMENTS
```

Return stdout exactly as produced.
