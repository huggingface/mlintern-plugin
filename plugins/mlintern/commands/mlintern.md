---
description: Run an ml-intern task
argument-hint: '[--background|--wait] [prompt]'
disable-model-invocation: true
allowed-tools: Bash(node:*)
---

Run:

```bash
node "${CLAUDE_PLUGIN_ROOT}/scripts/mlintern-companion.mjs" run $ARGUMENTS
```

Return stdout exactly as produced.
