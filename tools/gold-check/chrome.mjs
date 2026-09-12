// Driving headless Chrome over the DevTools protocol, which both adjudication checks need.
//
// Chrome is found through CHROME_PATH, or in the usual places on each platform.
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const CANDIDATES = [
  process.env.CHROME_PATH,
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
  "/usr/bin/google-chrome",
  "/usr/bin/google-chrome-stable",
  "/usr/bin/chromium",
  "/usr/bin/chromium-browser",
  "/snap/bin/chromium",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
];

export function browser() {
  const found = CANDIDATES.filter(Boolean).find((path) => existsSync(path));
  if (!found) throw new Error("no Chrome found. Set CHROME_PATH.");
  return found;
}

export const sleep = (ms) => new Promise((done) => setTimeout(done, ms));

export async function open(url, port) {
  const profile = await mkdtemp(join(tmpdir(), "steamgauge-gold-check-"));
  const chrome = spawn(
    browser(),
    [
      "--headless=new",
      `--remote-debugging-port=${port}`,
      `--user-data-dir=${profile}`,
      "--no-first-run",
      "--no-default-browser-check",
      "--disable-gpu",
      url,
    ],
    { stdio: "ignore" },
  );
  return {
    chrome,
    async close() {
      chrome.kill();
      // Chrome holds its profile open for a moment after the signal, and on Windows unlinking
      // a file it still has is an error rather than a wait.
      await new Promise((done) => chrome.once("exit", done));
      await rm(profile, { recursive: true, force: true }).catch(() => {});
    },
  };
}

export async function debuggerUrl(port) {
  for (let attempt = 0; attempt < 80; attempt += 1) {
    try {
      const list = await fetch(`http://127.0.0.1:${port}/json/list`).then((r) => r.json());
      const page = list.find((t) => t.type === "page" && t.webSocketDebuggerUrl);
      if (page) return page.webSocketDebuggerUrl;
    } catch {
      // Chrome has not opened the port yet, which is the usual case for the first second.
    }
    await sleep(250);
  }
  throw new Error("headless Chrome never opened a debugging port");
}

export async function connect(url) {
  const socket = new WebSocket(url);
  await new Promise((ok, bad) => {
    socket.addEventListener("open", ok, { once: true });
    socket.addEventListener("error", bad, { once: true });
  });
  let id = 0;
  const waiting = new Map();
  const asked = [];
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.method === "Network.requestWillBeSent") asked.push(message.params.request.url);
    const settle = waiting.get(message.id);
    if (settle) {
      waiting.delete(message.id);
      settle(message);
    }
  });
  const send = (method, params) =>
    new Promise((ok) => {
      id += 1;
      waiting.set(id, ok);
      socket.send(JSON.stringify({ id, method, params }));
    });
  const evaluate = (expression) =>
    send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
  return {
    socket,
    send,
    asked,
    evaluate,
    // The page renders its first question from a megabyte of embedded JSON, so "loaded" is not
    // the same moment as "answerable" and every step has to wait for the second one.
    async ready() {
      for (let attempt = 0; attempt < 80; attempt += 1) {
        const there = await evaluate(
          "document.readyState === 'complete' && !!document.querySelector('button.pick')",
        );
        if (there.result?.result?.value === true) return true;
        await sleep(250);
      }
      return false;
    },
  };
}
