#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { spawn, spawnSync } from "node:child_process";

import { binaryAvailable, processAlive, terminateProcess } from "./lib/process.mjs";
import {
  appendLog,
  generateJobId,
  listJobs,
  readJobFile,
  resolveJobLogFile,
  resolveWorkspaceRoot,
  upsertJob,
  writeJobFile
} from "./lib/state.mjs";

function printUsage() {
  process.stdout.write(
    [
      "Usage:",
      " node scripts/mlintern-companion.mjs setup [--json] [--cwd path]",
      " node scripts/mlintern-companion.mjs run [--background|--wait] [--model id] [--status [job-id]|--result [job-id]|--cancel [job-id]] [--json] [--cwd path] \"prompt\"",
      " node scripts/mlintern-companion.mjs worker --cwd path --job-id id",
      " node scripts/mlintern-companion.mjs status [job-id] [--json] [--cwd path]",
      " node scripts/mlintern-companion.mjs result [job-id] [--json] [--cwd path]",
      " node scripts/mlintern-companion.mjs cancel [job-id] [--json] [--cwd path]"
    ].join("\n")
  );
}

function parseArgs(argv) {
  const options = {};
  const positionals = [];
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith("--")) {
      positionals.push(token);
      continue;
    }
    const key = token.slice(2);
    if (["json", "background", "wait"].includes(key)) {
      options[key] = true;
      continue;
    }
    const next = argv[i + 1];
    if (next == null || next.startsWith("--")) {
      options[key] = true;
      continue;
    }
    options[key] = next;
    i += 1;
  }
  return { options, positionals };
}

function output(payload, rendered, asJson) {
  if (asJson) {
    process.stdout.write(`${JSON.stringify(payload, null, 2)}\n`);
    return;
  }
  process.stdout.write(rendered.endsWith("\n") ? rendered : `${rendered}\n`);
}

function firstLine(text, fallback) {
  const line = String(text || "")
    .split(/\r?\n/)
    .map((v) => v.trim())
    .find(Boolean);
  return line || fallback;
}

function chooseJob(cwd, reference = "") {
  const jobs = [...listJobs(cwd)].sort((a, b) => String(b.updatedAt || "").localeCompare(String(a.updatedAt || "")));
  if (!jobs.length) {
    throw new Error("No jobs found for this repository.");
  }
  if (!reference) {
    return jobs[0];
  }
  const exact = jobs.find((job) => job.id === reference);
  if (exact) {
    return exact;
  }
  const prefix = jobs.filter((job) => job.id.startsWith(reference));
  if (prefix.length === 1) {
    return prefix[0];
  }
  if (prefix.length > 1) {
    throw new Error(`Job reference "${reference}" is ambiguous.`);
  }
  throw new Error(`No job found for "${reference}".`);
}

function resolveModelOption(options) {
  if (!Object.hasOwn(options, "model")) {
    return null;
  }
  if (typeof options.model !== "string" || !options.model.trim()) {
    throw new Error("Provide a model id after --model. Example: --model huggingface/openai/gpt-oss-120b");
  }
  return options.model.trim();
}

function buildMlInternArgs(prompt, modelName) {
  const args = [];
  if (modelName) {
    args.push("--model", modelName);
  }
  args.push(prompt);
  return args;
}

function runMlInternSync(cwd, prompt, modelName = null) {
  const result = spawnSync("ml-intern", buildMlInternArgs(prompt, modelName), {
    cwd,
    encoding: "utf8",
    maxBuffer: 1024 * 1024 * 20
  });
  if (result.error) {
    return {
      status: 1,
      stdout: "",
      stderr: result.error.message || "Failed to start ml-intern."
    };
  }
  return {
    status: result.status ?? 1,
    stdout: result.stdout || "",
    stderr: result.stderr || ""
  };
}

async function runMlInternStreaming(cwd, prompt, modelName = null) {
  return await new Promise((resolve) => {
    const child = spawn("ml-intern", buildMlInternArgs(prompt, modelName), {
      cwd
    });
    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      const text = chunk.toString();
      stdout += text;
      process.stdout.write(text);
    });
    child.stderr.on("data", (chunk) => {
      const text = chunk.toString();
      stderr += text;
      process.stderr.write(text);
    });
    child.on("error", (error) => {
      const message = error?.message || "Failed to start ml-intern.";
      resolve({
        status: 1,
        stdout,
        stderr: stderr ? `${stderr}\n${message}` : message
      });
    });
    child.on("close", (code) => {
      resolve({
        status: code ?? 1,
        stdout,
        stderr
      });
    });
  });
}

function buildSetupPayload(cwd) {
  const workspaceRoot = resolveWorkspaceRoot(cwd);
  const nodeStatus = binaryAvailable("node", ["--version"], cwd);
  const mlInternStatus = binaryAvailable("ml-intern", ["--help"], cwd);
  const ready = nodeStatus.available && mlInternStatus.available;
  return {
    ready,
    workspaceRoot,
    node: nodeStatus,
    mlIntern: mlInternStatus,
    nextSteps: ready
      ? ["Run `/mlintern:run \"fine-tune a model\"`."]
      : ["Install ml-intern and ensure it is on PATH.", "Then rerun `/mlintern:setup`."]
  };
}

function renderSetup(payload) {
  return [
    "# ML Intern Setup",
    "",
    `Status: ${payload.ready ? "ready" : "needs attention"}`,
    "",
    `- node: ${payload.node.detail}`,
    `- ml-intern: ${payload.mlIntern.detail}`,
    "",
    "Next steps:",
    ...payload.nextSteps.map((step) => `- ${step}`)
  ].join("\n");
}

function spawnWorker(cwd, jobId) {
  const scriptPath = path.resolve(path.dirname(new URL(import.meta.url).pathname), "mlintern-companion.mjs");
  const child = spawn(process.execPath, [scriptPath, "worker", "--cwd", cwd, "--job-id", jobId], {
    cwd,
    env: process.env,
    detached: true,
    stdio: "ignore",
    windowsHide: true
  });
  child.unref();
  return child.pid ?? null;
}

function refreshActiveJob(cwd, job) {
  if (!["queued", "running"].includes(job.status)) {
    return job;
  }
  if (processAlive(job.pid)) {
    return job;
  }
  const stored = readJobFile(cwd, job.id);
  if (stored) {
    return stored;
  }
  const patched = {
    ...job,
    status: "failed",
    completedAt: new Date().toISOString(),
    pid: null,
    errorMessage: "Worker process exited unexpectedly."
  };
  upsertJob(cwd, patched);
  writeJobFile(cwd, job.id, patched);
  return patched;
}

function renderStatusJob(job) {
  const lines = [
    `- ${job.id} | ${job.status} | ${job.title || "ML Intern Task"}`,
    `  Summary: ${job.summary || ""}`
  ];
  if (job.threadId) {
    lines.push(`  Codex session: ${job.threadId}`);
  }
  if (job.logFile) {
    lines.push(`  Log: ${job.logFile}`);
  }
  if (job.errorMessage) {
    lines.push(`  Error: ${job.errorMessage}`);
  }
  return lines.join("\n");
}

async function handleSetup(argv) {
  const { options } = parseArgs(argv);
  const cwd = options.cwd ? path.resolve(process.cwd(), options.cwd) : process.cwd();
  const payload = buildSetupPayload(cwd);
  output(payload, renderSetup(payload), Boolean(options.json));
}

async function handleRun(argv) {
  const { options, positionals } = parseArgs(argv);
  const cwd = options.cwd ? path.resolve(process.cwd(), options.cwd) : process.cwd();
  const asJson = Boolean(options.json);
  const modelName = resolveModelOption(options);
  if (Object.hasOwn(options, "status")) {
    const reference = typeof options.status === "string" ? options.status : positionals[0] || "";
    await handleStatus(reference ? [reference, "--cwd", cwd, ...(asJson ? ["--json"] : [])] : ["--cwd", cwd, ...(asJson ? ["--json"] : [])]);
    return;
  }
  if (Object.hasOwn(options, "result")) {
    const reference = typeof options.result === "string" ? options.result : positionals[0] || "";
    await handleResult(reference ? [reference, "--cwd", cwd, ...(asJson ? ["--json"] : [])] : ["--cwd", cwd, ...(asJson ? ["--json"] : [])]);
    return;
  }
  if (Object.hasOwn(options, "cancel")) {
    const reference = typeof options.cancel === "string" ? options.cancel : positionals[0] || "";
    await handleCancel(reference ? [reference, "--cwd", cwd, ...(asJson ? ["--json"] : [])] : ["--cwd", cwd, ...(asJson ? ["--json"] : [])]);
    return;
  }
  const prompt = positionals.join(" ").trim();
  if (!prompt) {
    throw new Error("Provide a prompt. Example: /mlintern:run \"fine-tune a model\".");
  }
  const background = Boolean(options.background) && !options.wait;

  if (!background) {
    const result = asJson
      ? runMlInternSync(cwd, prompt, modelName)
      : await runMlInternStreaming(cwd, prompt, modelName);
    const payload = {
      status: result.status,
      stdout: result.stdout,
      stderr: result.stderr,
      model: modelName || null
    };
    if (asJson) {
      const rendered = result.stdout || result.stderr || "(no output)";
      output(payload, rendered, asJson);
    } else if (!result.stdout && !result.stderr) {
      process.stdout.write("(no output)\n");
    }
    if (result.status !== 0) {
      process.exitCode = result.status;
    }
    return;
  }

  const workspaceRoot = resolveWorkspaceRoot(cwd);
  const jobId = generateJobId();
  const logFile = resolveJobLogFile(cwd, jobId);
  const job = {
    id: jobId,
    workspaceRoot,
    title: "ML Intern Background Task",
    summary: firstLine(prompt, "ml-intern task"),
    status: "queued",
    prompt,
    model: modelName,
    logFile
  };
  writeJobFile(cwd, jobId, job);
  upsertJob(cwd, job);
  appendLog(cwd, jobId, "Queued for background execution.");
  const pid = spawnWorker(cwd, jobId);
  const queued = { ...job, pid, status: "running", startedAt: new Date().toISOString() };
  writeJobFile(cwd, jobId, queued);
  upsertJob(cwd, queued);
  appendLog(cwd, jobId, `Worker started with pid ${pid}.`);
  const payload = { jobId, status: "running", summary: queued.summary, model: modelName || null };
  const rendered = `ML Intern task started in background as ${jobId}. Check /mlintern:run --status ${jobId} for progress.`;
  output(payload, rendered, asJson);
}

async function handleWorker(argv) {
  const { options } = parseArgs(argv);
  const cwd = options.cwd ? path.resolve(process.cwd(), options.cwd) : process.cwd();
  const jobId = options["job-id"];
  if (!jobId) {
    throw new Error("Missing --job-id.");
  }
  const job = readJobFile(cwd, jobId);
  if (!job) {
    throw new Error(`Unknown job: ${jobId}`);
  }

  const started = { ...job, status: "running", startedAt: new Date().toISOString(), pid: process.pid };
  writeJobFile(cwd, jobId, started);
  upsertJob(cwd, started);
  appendLog(cwd, jobId, "Background execution started.");

  const child = spawn("ml-intern", buildMlInternArgs(job.prompt, job.model || null), { cwd });
  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (chunk) => {
    const text = chunk.toString();
    stdout += text;
    fs.appendFileSync(job.logFile, text, "utf8");
  });
  child.stderr.on("data", (chunk) => {
    const text = chunk.toString();
    stderr += text;
    fs.appendFileSync(job.logFile, text, "utf8");
  });

  const exitCode = await new Promise((resolve) => {
    child.on("close", (code) => resolve(code ?? 1));
    child.on("error", () => resolve(1));
  });

  const done = {
    ...started,
    status: exitCode === 0 ? "completed" : "failed",
    completedAt: new Date().toISOString(),
    pid: null,
    output: stdout,
    errorOutput: stderr,
    exitCode
  };
  if (exitCode !== 0) {
    done.errorMessage = firstLine(stderr, `ml-intern exited with code ${exitCode}`);
  }
  writeJobFile(cwd, jobId, done);
  upsertJob(cwd, done);
  appendLog(cwd, jobId, `Background execution finished with code ${exitCode}.`);
}

async function handleStatus(argv) {
  const { options, positionals } = parseArgs(argv);
  const cwd = options.cwd ? path.resolve(process.cwd(), options.cwd) : process.cwd();
  const asJson = Boolean(options.json);
  const reference = positionals[0] || "";
  if (reference) {
    const job = refreshActiveJob(cwd, chooseJob(cwd, reference));
    output(job, renderStatusJob(job), asJson);
    return;
  }
  const jobs = [...listJobs(cwd)]
    .sort((a, b) => String(b.updatedAt || "").localeCompare(String(a.updatedAt || "")))
    .slice(0, 10)
    .map((job) => refreshActiveJob(cwd, job));
  const payload = { jobs };
  const rendered = jobs.length
    ? ["# ML Intern Status", "", ...jobs.map((job) => renderStatusJob(job))].join("\n")
    : "# ML Intern Status\n\nNo jobs yet.";
  output(payload, rendered, asJson);
}

async function handleResult(argv) {
  const { options, positionals } = parseArgs(argv);
  const cwd = options.cwd ? path.resolve(process.cwd(), options.cwd) : process.cwd();
  const asJson = Boolean(options.json);
  const reference = positionals[0] || "";
  const job = refreshActiveJob(cwd, chooseJob(cwd, reference));
  if (["queued", "running"].includes(job.status)) {
    throw new Error(`Job ${job.id} is still ${job.status}. Run /mlintern:run --status ${job.id}.`);
  }
  const text = job.output || job.errorOutput || job.errorMessage || "No result output captured.";
  const payload = { job, output: text };
  output(payload, text, asJson);
}

async function handleCancel(argv) {
  const { options, positionals } = parseArgs(argv);
  const cwd = options.cwd ? path.resolve(process.cwd(), options.cwd) : process.cwd();
  const asJson = Boolean(options.json);
  const reference = positionals[0] || "";
  const job = chooseJob(cwd, reference);
  if (!["queued", "running"].includes(job.status)) {
    throw new Error(`Job ${job.id} is not running.`);
  }
  terminateProcess(job.pid);
  const cancelled = {
    ...job,
    status: "cancelled",
    completedAt: new Date().toISOString(),
    pid: null,
    errorMessage: "Cancelled by user."
  };
  writeJobFile(cwd, job.id, cancelled);
  upsertJob(cwd, cancelled);
  appendLog(cwd, job.id, "Cancelled by user.");
  const payload = { jobId: job.id, status: "cancelled" };
  output(payload, `Cancelled ${job.id}.`, asJson);
}

async function main() {
  const [subcommand, ...argv] = process.argv.slice(2);
  if (!subcommand || subcommand === "help" || subcommand === "--help") {
    printUsage();
    return;
  }
  if (subcommand === "setup") {
    await handleSetup(argv);
    return;
  }
  if (subcommand === "run") {
    await handleRun(argv);
    return;
  }
  if (subcommand === "worker") {
    await handleWorker(argv);
    return;
  }
  if (subcommand === "status") {
    await handleStatus(argv);
    return;
  }
  if (subcommand === "result") {
    await handleResult(argv);
    return;
  }
  if (subcommand === "cancel") {
    await handleCancel(argv);
    return;
  }
  throw new Error(`Unknown subcommand: ${subcommand}`);
}

main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
});
