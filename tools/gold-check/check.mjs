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
// `served.mjs` is the other half: this one asserts the page fetches nothing and keeps its own
// answers, which is what a file on disk has to do; that one asserts every answer reaches the
// server, which is what `--serve` promises and is how the page is meant to be run.
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

import { connect, debuggerUrl, open, sleep } from "./chrome.mjs";

const PORT = 9334;

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

// Every question in turn, on a page nobody has answered anything on. Marking by an offset
// counted in bytes and spent in a browser highlights most of a Chinese review instead of one
// sentence of it, and the first question in a fixture is usually English.
const MARKS = `(function () {
  var data = JSON.parse(document.getElementById('data').textContent);
  var wrong = [];
  // An HTML parser turns every carriage return in a text node into a newline, so a claim
  // written on Windows can never come back out of the DOM byte for byte. Both sides are put
  // in the same shape rather than the comparison being loosened to "close enough".
  var same = function (text) { return String(text).replace(/\\r\\n?/g, '\\n'); };
  for (var i = 0; i < data.questions.length; i += 1) {
    var shown = document.querySelector('.review mark');
    if (!shown) { wrong.push(i + ' has no mark at all'); break; }
    if (same(shown.textContent) !== same(data.questions[i].claim)) {
      wrong.push(i + ' marks ' + JSON.stringify(shown.textContent.slice(0, 30)) +
                 ' where the claim is ' + JSON.stringify(data.questions[i].claim.slice(0, 30)));
    }
    if (i + 1 < data.questions.length) {
      document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
    }
  }
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

// A browser loses its storage for reasons nobody controls, and by then the reader has spent a
// night on it. Emptying the store and reloading is that loss exactly: in-memory answers go with
// the page. What the reader has left is the file they exported, so it has to load back, and a
// file from another draw has to be refused rather than counted.
const EMPTIED = `(function () {
  var kept = Object.keys(localStorage).filter(function (k) { return k.indexOf('steamgauge-gold') === 0; });
  if (kept.length !== 1) return null;
  var name = kept[0];
  var held = JSON.parse(localStorage.getItem(name) || '{}');
  var rows = Object.keys(held).map(function (k) { return held[k]; })
    .filter(function (row) { return row && row.subject; });
  localStorage.removeItem(name);
  return { name: name, rows: rows };
})()`;

const restoring = (saved) => `(function () {
  var name = ${JSON.stringify(saved?.name ?? "")};
  var rows = ${JSON.stringify(saved?.rows ?? [])};
  if (!rows.length) { return Promise.resolve(['no answered claim to restore']); }
  var input = document.getElementById('restore');
  if (!input) { return Promise.resolve(['the page offers no way to load answers back']); }

  var stranger = { app_id: 999999, review_id: 'not-in-this-draw', index: 0, subject: 'price',
                   polarity: 'praise', ambiguous: false, split_wrong: false, unsure: false };
  var transfer = new DataTransfer();
  transfer.items.add(new File([JSON.stringify(rows.concat([stranger]))], 'gold-answers.json',
                              { type: 'application/json' }));
  input.files = transfer.files;
  input.dispatchEvent(new Event('change', { bubbles: true }));

  return new Promise(function (done) {
    setTimeout(function () {
      var wrong = [];
      var back = JSON.parse(localStorage.getItem(name) || '{}');
      if (Object.keys(back).length !== rows.length) {
        wrong.push('loading an exported file back restored ' + Object.keys(back).length +
                   ' of ' + rows.length + ' answers');
      }
      if (back['999999#not-in-this-draw#0']) {
        wrong.push('an answer from another draw was taken into this one');
      }
      var said = document.querySelector('.note.loaded');
      if (!said) {
        wrong.push('the page does not say what it loaded');
      } else if (said.textContent.indexOf('ignored') === -1) {
        wrong.push('the page does not say it ignored an answer from another draw');
      }
      done(wrong);
    }, 400);
  });
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

const chrome = await open(page, PORT);

let failed = true;
try {
  const { socket, send, asked, evaluate, ready } = await connect(await debuggerUrl(PORT));
  await send("Network.enable", {});
  await send("Page.reload", { ignoreCache: true });
  await ready();

  const fetched = asked.filter((url) => url !== page);

  // Walked first, on a page nobody has answered anything on, because answering moves which
  // question is on screen and the marks have to be checked against all of them.
  const marks = (await evaluate(MARKS)).result?.result?.value ?? ["the marks could not be read"];
  await send("Page.reload", { ignoreCache: true });
  await ready();

  const answer = await evaluate(PROBE);
  const wrong = [
    ...marks,
    ...(answer.result?.result?.value ?? ["the page could not be driven at all"]),
  ];

  // Reloading is what a reader does after closing the tab, and it is the whole reason the
  // answers are kept at all.
  await send("Page.reload", { ignoreCache: true });
  await ready();
  const again = (await evaluate(RELOADED)).result?.result?.value ?? [];

  const saved = (await evaluate(EMPTIED)).result?.result?.value ?? null;
  await send("Page.reload", { ignoreCache: true });
  await ready();
  const restored = saved
    ? ((await evaluate(restoring(saved))).result?.result?.value ?? [
        "loading answers back could not be driven",
      ])
    : ["nothing was stored to restore from"];

  await send("Emulation.setDeviceMetricsOverride", {
    width: 420,
    height: 900,
    deviceScaleFactor: 1,
    mobile: true,
  });
  await sleep(250);
  const narrow = (await evaluate(NARROW)).result?.result?.value ?? [];

  const all = [...wrong, ...again, ...restored, ...narrow];
  if (fetched.length) all.push(`the page fetched ${fetched.length}: ${fetched.join(", ")}`);

  if (all.length) {
    console.error(`${file}\n  ${all.join("\n  ")}`);
  } else {
    console.log(`${file}: adjudication page keeps every promise it makes`);
    failed = false;
  }
  socket.close();
} finally {
  await chrome.close();
}

process.exit(failed ? 1 : 0);
