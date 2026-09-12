// Drives the adjudication page as it is actually served, and fails if an answer does not reach
// the disk.
//
// `check.mjs` opens the page as a file, where the browser is the only copy. `steamgauge gold
// --serve` is the way the page is meant to be run, and there the promise is stronger: every
// answer is on disk before the next question is drawn, and reopening the page anywhere carries
// on from the file rather than from this browser's storage. That promise has a failure mode
// nobody would notice until it mattered, because a POST that silently does nothing looks
// exactly like one that worked.
//
// The server here stands in for the Rust one, which is unit tested separately. What is being
// checked is the page's half of the contract.
//
//   cargo run -p steamgauge-core --example sample-gold -- gold.html
//   node tools/gold-check/served.mjs gold.html
import { readFile } from "node:fs/promises";
import { createServer } from "node:http";
import { resolve } from "node:path";

import { connect, debuggerUrl, open, sleep } from "./chrome.mjs";

const DEBUG_PORT = 9336;

const file = resolve(process.argv[2] ?? "gold.html");
const page = await readFile(file, "utf8");

let held = [];
let posts = 0;
const server = createServer((request, reply) => {
  const path = (request.url ?? "/").split("?")[0];
  if (request.method === "GET" && path === "/") {
    reply.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
    reply.end(page);
    return;
  }
  if (request.method === "GET" && path === "/answers") {
    reply.writeHead(200, { "Content-Type": "application/json" });
    reply.end(JSON.stringify(held));
    return;
  }
  if (request.method === "POST" && path === "/answers") {
    let body = "";
    request.on("data", (chunk) => {
      body += chunk;
    });
    request.on("end", () => {
      try {
        held = JSON.parse(body);
        posts += 1;
      } catch {
        // A body that is not JSON is a failure the assertions below will report as a missing
        // answer, which is the shape the reader would see.
      }
      reply.writeHead(200, { "Content-Type": "application/json" });
      reply.end('{"saved":true}');
    });
    return;
  }
  reply.writeHead(404).end();
});
await new Promise((ok) => server.listen(0, "127.0.0.1", ok));
const served = `http://127.0.0.1:${server.address().port}/`;

// Answers the first question, the way a reader does.
const ANSWER = `(function () {
  var press = function (key) {
    document.dispatchEvent(new KeyboardEvent('keydown', { key: key, bubbles: true }));
  };
  var first = document.querySelector('button.pick');
  press(first.dataset.key || first.textContent.trim().charAt(0));
  press('2');
  return document.getElementById('kept') ? document.getElementById('kept').textContent : 'no indicator';
})()`;

const FORGET = `(function () {
  Object.keys(localStorage)
    .filter(function (k) { return k.indexOf('steamgauge-gold') === 0; })
    .forEach(function (k) { localStorage.removeItem(k); });
  return true;
})()`;

const ANSWERED = `(function () {
  var counter = document.querySelector('.count');
  return counter ? counter.textContent : '';
})()`;

const chrome = await open(served, DEBUG_PORT);
let failed = true;
try {
  const { socket, send, evaluate, ready } = await connect(await debuggerUrl(DEBUG_PORT));
  await send("Network.enable", {});
  if (!(await ready())) throw new Error("the served page never became answerable");

  const wrong = [];
  const kept = (await evaluate(ANSWER)).result?.result?.value;
  await sleep(500);

  if (posts === 0) wrong.push("answering posted nothing, so the answer only exists in the browser");
  if (!held.length) wrong.push("the post carried no answers");
  if (held.length && !held[0].subject) wrong.push("the posted answer carries no subject");
  if (held.length && held[0].polarity !== "complaint") {
    wrong.push(`the posted answer says polarity ${held[0].polarity}, not the one pressed`);
  }
  if (kept !== "saved") wrong.push(`the header says ${JSON.stringify(kept)} rather than saved`);

  // The promise that matters: a browser that remembers nothing still picks up the work,
  // because the file is the copy and this is only a cache.
  await evaluate(FORGET);
  await send("Page.reload", { ignoreCache: true });
  if (!(await ready())) throw new Error("the served page never became answerable again");
  await sleep(500);
  const counter = (await evaluate(ANSWERED)).result?.result?.value ?? "";
  if (!/\b1 answered\b/.test(counter)) {
    wrong.push(`a browser with no storage read back ${JSON.stringify(counter)}, not 1 answered`);
  }

  if (wrong.length) {
    console.error(`${file}\n  ${wrong.join("\n  ")}`);
  } else {
    console.log(`${file}: served, every answer reaches the file before the next question`);
    failed = false;
  }
  socket.close();
} finally {
  await chrome.close();
  server.close();
}

process.exit(failed ? 1 : 0);
