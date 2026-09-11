// Drives the adjudication page in headless Chrome and fails if it does not do what it promises.
//
// This page is the only place a person's judgement enters the project, and every figure that
// can be called accuracy rather than agreement will rest on it. A keyboard shortcut that does
// nothing, an answer that does not survive a reload, or an export that drops a field would all
// be discovered after somebody had read a thousand claims, which is exactly too late.
//
//   steamgauge gold --to gold.html
//   node tools/gold-check/check.mjs gold.html
//
// Chrome is found through CHROME_PATH, or in the usual places on each platform.
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const PORT = 9334;

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

function browser() {
  const found = CANDIDATES.filter(Boolean).find((path) => existsSync(path));
  if (!found) throw new Error("no Chrome found. Set CHROME_PATH.");
  return found;
}

const sleep = (ms) => new Promise((done) => setTimeout(done, ms));

async function debuggerUrl() {
  for (let attempt = 0; attempt < 80; attempt += 1) {
    try {
      const list = await fetch(`http://127.0.0.1:${PORT}/json/list`).then((r) => r.json());
      const page = list.find((t) => t.type === "page" && t.webSocketDebuggerUrl);
      if (page) return page.webSocketDebuggerUrl;
    } catch {
      // Chrome has not opened the port yet, which is the usual case for the first second.
    }
    await sleep(250);
  }
  throw new Error("headless Chrome never opened a debugging port");
}

async function connect(url) {
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
  return {
    socket,
    send,
    asked,
    evaluate: (expression) =>
      send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true }),
  };
}

// Runs inside the page. Returns a list of failures, so one run reports everything wrong.
const PROBE = `(function () {
  var wrong = [];
  var check = function (claim, ok) { if (!ok) { wrong.push(claim); } };
  var press = function (key) {
    document.dispatchEvent(new KeyboardEvent('keydown', { key: key, bubbles: true }));
  };
  // Which claim is on screen, and nothing else: the counter beside it also carries how many
  // have been answered, which moves whenever an answer is given without the page advancing.
  var at = function () {
    return Number(/^(\\d+) of/.exec(document.querySelector('header.bar .count').textContent)[1]);
  };

  var data = JSON.parse(document.getElementById('data').textContent);
  check('the page carries no questions', data.questions && data.questions.length > 0);
  check('the page carries no category sheet', data.categories && data.categories.length > 20);

  check('nothing is rendered', document.querySelectorAll('button.pick').length > 0);

  // Every category must be reachable from the keyboard, and by its own key. Two categories
  // sharing one means the reader finds out at claim four hundred that half the sheet cannot
  // be picked without the mouse.
  var shortcuts = Array.prototype.map.call(
    document.querySelectorAll('button.pick kbd'), function (k) { return k.textContent; }
  );
  check('a category has no keyboard shortcut', shortcuts.every(function (k) { return k.length === 1; }));
  check('two categories share a keyboard shortcut',
    new Set(shortcuts).size === shortcuts.length);
  check('there are fewer shortcuts than categories', shortcuts.length === data.categories.length);

  // And the key must pick the category it is printed on, not merely some category.
  var buttons = document.querySelectorAll('button.pick');
  var wanted = buttons[buttons.length - 1];
  var wantedId = wanted.getAttribute('data-subject');
  press(wanted.querySelector('kbd').textContent);
  var chosen = document.querySelector('button.pick.chosen');
  check('a shortcut picks a different category from the one it is printed on',
    chosen && chosen.getAttribute('data-subject') === wantedId);
  check('the sheet is not on the page to consult',
    document.querySelectorAll('details.sheet dt').length === data.categories.length);

  // The claim must be marked inside its review, once, and the marked text must be the claim.
  var mark = document.querySelector('.review mark');
  check('the claim is not marked inside its review', !!mark);
  if (mark) {
    check('the mark is not the claim itself',
      mark.textContent.trim() === data.questions[0].claim.trim());
  }

  // A blind question must show no answer.
  if (!data.questions[0].shown) {
    check('a blind question shows an answer', document.querySelectorAll('.shown').length === 0);
  }

  // Picking a subject alone must not advance: half an answer is not an answer.
  var first = at();
  document.querySelector('button.pick').click();
  check('picking a subject alone advances the page', at() === first);
  check('picking a subject does not mark it chosen',
    document.querySelectorAll('button.pick.chosen').length === 1);

  // Adding a polarity completes it and must advance.
  document.querySelector('button.tone[data-tone="praise"]').click();
  check('a complete answer does not advance the page', at() === first + 1);

  // The keyboard must do the same work as the mouse.
  var second = at();
  var letter = document.querySelector('button.pick kbd').textContent;
  press(letter);
  press('1');
  check('the keyboard does not answer a claim', at() === second + 1);

  press('ArrowLeft');
  check('the left arrow does not go back', at() === second);

  // An answer already given must come back when the claim does.
  check('an answered claim does not show its answer again',
    document.querySelectorAll('button.pick.chosen').length === 1 &&
    document.querySelectorAll('button.tone.chosen').length >= 1);

  // Everything answered must be kept, so a closed tab does not cost a night.
  var kept = Object.keys(localStorage).filter(function (k) { return k.indexOf('steamgauge-gold') === 0; });
  check('nothing is kept for the next sitting', kept.length === 1);
  var held = JSON.parse(localStorage.getItem(kept[0]) || '{}');
  check('fewer answers were kept than were given', Object.keys(held).length >= 2);
  var one = held[Object.keys(held)[0]];
  ['app_id', 'review_id', 'index', 'subject', 'polarity'].forEach(function (field) {
    check('a kept answer has no ' + field, one[field] !== undefined);
  });

  check('the progress bar never moves',
    parseFloat(document.querySelector('.progress i').style.width) > 0);

  // A reader who opens the sheet to settle a boundary must not have it shut on them by the
  // act of answering, which is the one moment they were reading it for.
  var sheet = document.querySelector('details.sheet');
  sheet.open = true;
  sheet.dispatchEvent(new Event('toggle'));
  document.querySelector('button.pick').click();
  document.querySelector('button.tone[data-tone="neutral"]').click();
  check('the category sheet shuts itself when a claim is answered',
    document.querySelector('details.sheet').open === true);

  return wrong;
})()`;

const RELOADED = `(function () {
  var wrong = [];
  var kept = Object.keys(localStorage).filter(function (k) { return k.indexOf('steamgauge-gold') === 0; });
  if (kept.length !== 1) { return ['answers did not survive a reload']; }
  var held = JSON.parse(localStorage.getItem(kept[0]) || '{}');
  if (Object.keys(held).length < 2) { wrong.push('answers were lost across a reload'); }
  var shown = document.querySelector('header.bar .count');
  if (!shown || shown.textContent.indexOf('answered') === -1) {
    wrong.push('the page does not say how much is answered');
  }
  if (shown && /^1 of /.test(shown.textContent)) {
    wrong.push('a reload starts again at the first claim rather than the first unanswered one');
  }
  return wrong;
})()`;

const NARROW = `(function () {
  var wrong = [];
  if (document.documentElement.scrollWidth > window.innerWidth + 1) {
    wrong.push('the page scrolls sideways on a phone');
  }
  var claim = document.querySelector('.claim');
  if (claim && claim.getBoundingClientRect().right > window.innerWidth + 1) {
    wrong.push('the claim runs off the side of a phone');
  }
  return wrong;
})()`;

const file = resolve(process.argv[2] ?? "gold.html");
const page = pathToFileURL(file).href;
const profile = await mkdtemp(join(tmpdir(), "steamgauge-gold-check-"));

const chrome = spawn(
  browser(),
  [
    "--headless=new",
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${profile}`,
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-gpu",
    page,
  ],
  { stdio: "ignore" },
);

let failed = true;
try {
  const { socket, send, asked, evaluate } = await connect(await debuggerUrl());
  await send("Network.enable", {});
  await send("Page.reload", { ignoreCache: true });
  for (let attempt = 0; attempt < 80; attempt += 1) {
    const ready = await evaluate("document.readyState === 'complete' && !!document.querySelector('button.pick')");
    if (ready.result?.result?.value === true) break;
    await sleep(250);
  }

  const fetched = asked.filter((url) => url !== page);
  const answer = await evaluate(PROBE);
  const wrong = answer.result?.result?.value ?? ["the page could not be driven at all"];

  // Reloading is what a reader does after closing the tab, and it is the whole reason the
  // answers are kept at all.
  await send("Page.reload", { ignoreCache: true });
  for (let attempt = 0; attempt < 80; attempt += 1) {
    const ready = await evaluate("document.readyState === 'complete' && !!document.querySelector('button.pick')");
    if (ready.result?.result?.value === true) break;
    await sleep(250);
  }
  const again = (await evaluate(RELOADED)).result?.result?.value ?? [];

  await send("Emulation.setDeviceMetricsOverride", {
    width: 420,
    height: 900,
    deviceScaleFactor: 1,
    mobile: true,
  });
  await sleep(250);
  const narrow = (await evaluate(NARROW)).result?.result?.value ?? [];

  const all = [...wrong, ...again, ...narrow];
  if (fetched.length) all.push(`the page fetched ${fetched.length}: ${fetched.join(", ")}`);

  if (all.length) {
    console.error(`${file}\n  ${all.join("\n  ")}`);
  } else {
    console.log(`${file}: adjudication page keeps every promise it makes`);
    failed = false;
  }
  socket.close();
} finally {
  chrome.kill();
  // Chrome holds its profile open for a moment after the signal, and on Windows unlinking a
  // file it still has is an error rather than a wait.
  await new Promise((done) => chrome.once("exit", done));
  await rm(profile, { recursive: true, force: true }).catch(() => {});
}

process.exit(failed ? 1 : 0);
