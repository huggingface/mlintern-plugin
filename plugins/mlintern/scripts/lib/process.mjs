import { spawnSync } from "node:child_process";

export function binaryAvailable(command, args = ["--version"], cwd = process.cwd()) {
  const result = spawnSync(command, args, { cwd, encoding: "utf8" });
  if (result.status === 0) {
    const version = (result.stdout || result.stderr || "").trim();
    return { available: true, detail: version || `${command} available` };
  }
  return { available: false, detail: `${command} not available` };
}

export function processAlive(pid) {
  if (!Number.isFinite(pid)) {
    return false;
  }
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

export function terminateProcess(pid) {
  if (!Number.isFinite(pid)) {
    return false;
  }
  try {
    process.kill(pid, "SIGTERM");
    return true;
  } catch {
    return false;
  }
}
