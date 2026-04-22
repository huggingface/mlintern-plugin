import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";

const PLUGIN_DATA_ENV = "CLAUDE_PLUGIN_DATA";
const FALLBACK_ROOT = path.join(os.tmpdir(), "mlintern-plugin");
const STATE_FILE = "state.json";
const JOBS_DIR = "jobs";

function nowIso() {
  return new Date().toISOString();
}

function defaultState() {
  return {
    version: 1,
    jobs: []
  };
}

export function resolveWorkspaceRoot(cwd) {
  const result = spawnSync("git", ["rev-parse", "--show-toplevel"], {
    cwd,
    encoding: "utf8"
  });
  if (result.status === 0 && result.stdout.trim()) {
    return result.stdout.trim();
  }
  return cwd;
}

export function resolveStateDir(cwd) {
  const workspace = resolveWorkspaceRoot(cwd);
  const slug = path.basename(workspace).replace(/[^a-zA-Z0-9._-]+/g, "-") || "workspace";
  const hash = createHash("sha256").update(workspace).digest("hex").slice(0, 12);
  const root = process.env[PLUGIN_DATA_ENV]
    ? path.join(process.env[PLUGIN_DATA_ENV], "state")
    : FALLBACK_ROOT;
  return path.join(root, `${slug}-${hash}`);
}

export function resolveJobsDir(cwd) {
  return path.join(resolveStateDir(cwd), JOBS_DIR);
}

export function resolveStateFile(cwd) {
  return path.join(resolveStateDir(cwd), STATE_FILE);
}

export function resolveJobFile(cwd, jobId) {
  return path.join(resolveJobsDir(cwd), `${jobId}.json`);
}

export function resolveJobLogFile(cwd, jobId) {
  return path.join(resolveJobsDir(cwd), `${jobId}.log`);
}

export function ensureStateDir(cwd) {
  fs.mkdirSync(resolveJobsDir(cwd), { recursive: true });
}

export function loadState(cwd) {
  const file = resolveStateFile(cwd);
  if (!fs.existsSync(file)) {
    return defaultState();
  }
  try {
    const parsed = JSON.parse(fs.readFileSync(file, "utf8"));
    return {
      ...defaultState(),
      ...parsed,
      jobs: Array.isArray(parsed.jobs) ? parsed.jobs : []
    };
  } catch {
    return defaultState();
  }
}

export function saveState(cwd, state) {
  ensureStateDir(cwd);
  fs.writeFileSync(resolveStateFile(cwd), `${JSON.stringify(state, null, 2)}\n`, "utf8");
  return state;
}

export function updateState(cwd, mutate) {
  const state = loadState(cwd);
  mutate(state);
  return saveState(cwd, state);
}

export function generateJobId() {
  return `mli-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

export function upsertJob(cwd, patch) {
  const timestamp = nowIso();
  return updateState(cwd, (state) => {
    const idx = state.jobs.findIndex((job) => job.id === patch.id);
    if (idx < 0) {
      state.jobs.unshift({ createdAt: timestamp, updatedAt: timestamp, ...patch });
      return;
    }
    state.jobs[idx] = { ...state.jobs[idx], ...patch, updatedAt: timestamp };
  });
}

export function listJobs(cwd) {
  return loadState(cwd).jobs;
}

export function readJobFile(cwd, jobId) {
  const file = resolveJobFile(cwd, jobId);
  if (!fs.existsSync(file)) {
    return null;
  }
  return JSON.parse(fs.readFileSync(file, "utf8"));
}

export function writeJobFile(cwd, jobId, payload) {
  ensureStateDir(cwd);
  fs.writeFileSync(resolveJobFile(cwd, jobId), `${JSON.stringify(payload, null, 2)}\n`, "utf8");
}

export function appendLog(cwd, jobId, message) {
  if (!message) {
    return;
  }
  const logFile = resolveJobLogFile(cwd, jobId);
  fs.appendFileSync(logFile, `[${nowIso()}] ${String(message).trim()}\n`, "utf8");
}
