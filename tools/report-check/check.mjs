// Drives a rendered report in headless Chrome and fails if its scripting does not do what
// the page promises.
//
// The report is one self-contained file whose folding, filtering, sorting and deep links are
// the only part of this tool no Rust test can reach: they exist only once a browser has run
// them. Everything checked here is a promise the page makes in its own prose.
//
//   cargo run -p census-core --example sample-report -- page.html
//   node tools/report-check/check.mjs page.html
//
// Chrome is found through CHROME_PATH, or in the usual places on each platform.
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const PORT = 9333;

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
  if (!found) {
    throw new Error(
      `no Chrome found. Set CHROME_PATH, or install one of:\n  ${CANDIDATES.filter(Boolean).join("\n  ")}`,
    );
  }
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
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    const settle = waiting.get(message.id);
    if (settle) {
      waiting.delete(message.id);
      settle(message);
    }
  });
  return {
    socket,
    evaluate(expression) {
      return new Promise((ok) => {
        id += 1;
        waiting.set(id, ok);
        socket.send(
          JSON.stringify({
            id,
            method: "Runtime.evaluate",
            params: { expression, returnByValue: true, awaitPromise: true },
          }),
        );
      });
    },
  };
}

// Runs inside the page. Returns a list of failures, so one run reports everything wrong
// rather than the first thing wrong.
const PROBE = `(function () {
  var wrong = [];
  var check = function (claim, ok) { if (!ok) { wrong.push(claim); } };
  var visible = function (selector) {
    return Array.prototype.filter.call(document.querySelectorAll(selector), function (el) {
      return !el.classList.contains('filtered-out');
    }).length;
  };

  check('the stylesheet is never told scripting works', document.documentElement.classList.contains('js'));

  var panels = document.querySelectorAll('table.categories tbody tr.panel');
  check('there is no evidence to fold', panels.length > 0);
  check('panels do not start folded', Array.prototype.every.call(panels, function (p) { return p.hidden; }));

  // Filtering.
  var form = document.querySelector('[data-filter]');
  var input = document.querySelector('[data-filter-input]');
  var count = document.querySelector('[data-filter-count]');
  check('the filter is never revealed', form && form.hidden === false);
  if (input && count) {
    var everything = count.textContent;
    var rows = visible('table.categories tbody tr.row');
    var matrix = visible('table.matrix tbody tr');
    check('nothing is on screen to filter', rows > 0 && matrix > 0);

    input.value = 'price';
    input.dispatchEvent(new Event('input'));
    check('filtering narrows no category table', visible('table.categories tbody tr.row') < rows);
    check('filtering narrows no cross-game matrix', visible('table.matrix tbody tr') < matrix);
    check('a filtered row keeps its panel on screen', (function () {
      return Array.prototype.every.call(panels, function (p) {
        var row = p.previousElementSibling;
        return !row || row.classList.contains('filtered-out') === p.classList.contains('filtered-out');
      });
    })());
    var left = Array.prototype.filter.call(
      document.querySelectorAll('table.categories tbody tr.row'),
      function (r) { return !r.classList.contains('filtered-out'); }
    );
    check('a filter keeps rows that do not match it', left.every(function (r) {
      return r.getAttribute('data-name').indexOf('price') !== -1;
    }));

    input.value = 'zzzzzz';
    input.dispatchEvent(new Event('input'));
    check('a filter matching nothing leaves rows on screen', visible('table.categories tbody tr.row') === 0);
    check('a filter matching nothing says nothing', count.textContent.length > 0);

    input.value = '';
    input.dispatchEvent(new Event('input'));
    check('clearing the filter loses rows', visible('table.categories tbody tr.row') === rows);
    check('clearing the filter rewords the count', count.textContent === everything);
  }

  // Folding.
  var row = document.querySelector('tr.row[data-expands]');
  var panel = row && document.getElementById(row.getAttribute('data-expands'));
  if (panel) {
    row.click();
    check('opening a row shows nothing', panel.hidden === false);
    check('an opened row does not say so', row.getAttribute('aria-expanded') === 'true');
    check('opening one row opens others', Array.prototype.filter.call(panels, function (p) {
      return !p.hidden;
    }).length === 1);
    row.click();
    check('a row cannot be closed again', panel.hidden === true);
  }

  // Sorting, which has to carry each panel along with the row it belongs to.
  var table = document.querySelector('table.categories');
  var button = table && table.querySelector('thead th.num button.sort');
  if (button) {
    button.click();
    var values = [];
    var paired = true;
    Array.prototype.forEach.call(table.tBodies[0].rows, function (r) {
      if (!r.classList.contains('row')) { return; }
      var id = r.getAttribute('data-expands');
      if (id && (!r.nextElementSibling || r.nextElementSibling.id !== id)) { paired = false; }
      var cell = r.children[1];
      if (cell && cell.getAttribute('data-value') !== null) {
        values.push(parseFloat(cell.getAttribute('data-value')));
      }
    });
    check('sorting separates a row from its own evidence', paired);
    check('sorting a number column does not sort it', values.every(function (v, i) {
      return i === 0 || values[i - 1] >= v;
    }));
    check('a sorted column does not say which way', button.parentNode.getAttribute('aria-sort') === 'descending');
  }

  // A link from the finding to the reviews behind it.
  var link = document.querySelector('.headline a[href^="#panel-"]');
  check('the finding leads nowhere', Boolean(link));
  if (link) {
    var target = document.getElementById(link.getAttribute('href').slice(1));
    check('the finding points at nothing', Boolean(target));
    if (target) {
      window.location.hash = link.getAttribute('href');
      window.dispatchEvent(new HashChangeEvent('hashchange'));
      check('a link into the evidence lands on a closed row', target.hidden === false);
      check('a row opened by a link does not say so',
        target.previousElementSibling.getAttribute('aria-expanded') === 'true');
    }
  }

  // Printing: every fold open, every filtered row still gone.
  if (input) {
    input.value = 'price';
    input.dispatchEvent(new Event('input'));
  }
  var pile = document.querySelector('details.pile');
  if (pile) {
    pile.open = false;
    window.dispatchEvent(new Event('beforeprint'));
    check('printing leaves a fold shut', pile.open === true);
    check('printing undoes the reader\\'s filter', document.querySelectorAll('.filtered-out').length > 0);
    window.dispatchEvent(new Event('afterprint'));
    check('printing leaves the page unfolded afterwards', pile.open === false);
  }

  // The theme control has to actually stamp a choice on the page.
  var theme = document.querySelector('[data-theme-toggle]');
  if (theme) {
    theme.click();
    var chosen = document.documentElement.getAttribute('data-theme');
    check('the theme control chooses nothing', chosen === 'dark' || chosen === 'light');
    check('the theme control does not say what it chose',
      theme.getAttribute('aria-pressed') === String(chosen === 'dark'));
  }

  return wrong;
})()`;

const page = pathToFileURL(resolve(process.argv[2] ?? "sample-report.html")).href;
const profile = await mkdtemp(join(tmpdir(), "census-report-check-"));
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
  const { socket, evaluate } = await connect(await debuggerUrl());
  for (let attempt = 0; attempt < 80; attempt += 1) {
    const ready = await evaluate(
      "document.readyState === 'complete' && document.documentElement.classList.contains('js')",
    );
    if (ready.result?.result?.value === true) break;
    await sleep(250);
  }

  const answer = await evaluate(PROBE);
  socket.close();
  if (answer.result?.exceptionDetails) {
    console.error("the page threw while being checked:");
    console.error(answer.result.exceptionDetails.exception?.description ?? answer.result.exceptionDetails.text);
  } else {
    const wrong = answer.result.result.value;
    if (wrong.length === 0) {
      console.log(`the report behaves as it says it does: ${page}`);
      failed = false;
    } else {
      console.error(`${wrong.length} promise(s) the page does not keep:`);
      for (const claim of wrong) console.error(`  - ${claim}`);
    }
  }
} finally {
  chrome.kill();
  await rm(profile, { recursive: true, force: true }).catch(() => {});
}
process.exit(failed ? 1 : 0);
