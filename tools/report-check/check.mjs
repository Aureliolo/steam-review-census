// Drives a rendered report in headless Chrome and fails if it does not do what it promises.
//
// The report is one self-contained file whose folding, filtering, sorting and deep links
// exist only once a browser has run them, and whose rendering only exists once a browser has
// laid it out: a stylesheet rule can flatten a chart, put half of every review off the side
// of a phone or turn a printed page into blocks of ink while the markup stays exactly right.
// Both are the part of this tool no Rust test can reach.
//
// The same page is driven four times: on a desktop window, at 420 pixels, under print media
// with the machine asking for a dark screen, and again with the script cut out, which is what
// a reader with scripting off is served. Everything checked is a promise the page makes.
//
//   cargo run -p census-core --example sample-report -- page.html
//   node tools/report-check/check.mjs page.html
//
// Chrome is found through CHROME_PATH, or in the usual places on each platform.
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
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
  // Every URL the page asks for, which is the only way to hold it to fetching nothing: the
  // markup can be free of every http:// this tool would have written and still pull a font
  // in from a stylesheet nobody read closely.
  const asked = [];
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.method === "Network.requestWillBeSent") {
      asked.push(message.params.request.url);
    }
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
    // Only a report of several games has one, and a report of one game must still work.
    var matrix = visible('table.matrix tbody tr');
    check('nothing is on screen to filter', rows > 0);

    input.value = 'price';
    input.dispatchEvent(new Event('input'));
    check('filtering narrows no category table', visible('table.categories tbody tr.row') < rows);
    check('filtering narrows no cross-game matrix',
      matrix === 0 || visible('table.matrix tbody tr') < matrix);
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

  // Folding, whose control has to be a real button inside a row that stays a row.
  var row = document.querySelector('tr.row[data-expands]');
  var panel = row && document.getElementById(row.getAttribute('data-expands'));
  var button = row && row.querySelector('button.disclose');
  check('no row has a control to open it', Boolean(button));
  if (panel && button) {
    check('a row was given a role that stops it being a row', row.getAttribute('role') === null);
    check('the control does not say what it opens',
      button.getAttribute('aria-controls') === panel.id);
    check('the control does not start shut', button.getAttribute('aria-expanded') === 'false');
    check('the control is empty', button.textContent.trim().length > 0);

    row.click();
    check('opening a row shows nothing', panel.hidden === false);
    check('an opened row does not say so', button.getAttribute('aria-expanded') === 'true');
    check('opening one row opens others', Array.prototype.filter.call(panels, function (p) {
      return !p.hidden;
    }).length === 1);
    // The panel is a cell in a table that scrolls sideways, so its content is laid out at the
    // width of the table unless something says otherwise, and on a narrow screen that puts
    // most of every quoted review off the side of the page.
    var quoted = panel.querySelector('.text p') || panel.querySelector('p');
    if (quoted) {
      check('a quoted review is laid out wider than the screen it is read on',
        Math.round(quoted.getBoundingClientRect().width) <= document.documentElement.clientWidth);
    }

    row.click();
    check('a row cannot be closed again', panel.hidden === true);

    // Selecting a figure out of the table must not fold the row it was read from.
    var selection = window.getSelection();
    var range = document.createRange();
    range.selectNodeContents(row);
    selection.removeAllRanges();
    selection.addRange(range);
    row.click();
    check('copying a number out of a row folds it away', panel.hidden === true);
    selection.removeAllRanges();

    // Reaching the control by keyboard has to open it exactly once.
    button.focus();
    check('the control cannot be reached by keyboard', document.activeElement === button);
    button.click();
    check('activating the control does not open the row', panel.hidden === false);
    button.click();
    check('activating the control twice does not close the row', panel.hidden === true);
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

    // The name cell also holds a warning mark and a sentence for screen readers, so sorting
    // by what is written in it is not the same as sorting by the category's name.
    var byName = table.querySelector('thead th:not(.num) button.sort');
    if (byName) {
      byName.click();
      var names = [];
      Array.prototype.forEach.call(table.tBodies[0].rows, function (r) {
        if (r.classList.contains('row')) { names.push(r.getAttribute('data-name')); }
      });
      check('sorting by category does not sort by category name', names.every(function (n, i) {
        return i === 0 || names[i - 1] <= n;
      }));
    }
  }

  // Wide things belong in the containers built to scroll them. Anything reaching past the
  // page pans the whole document sideways, which on a phone moves every column of text on
  // every line, and nothing in the markup shows it.
  check('the page itself pans sideways',
    document.documentElement.scrollWidth <= document.documentElement.clientWidth);

  // The only links off this page are the ones back to the review on Steam, and a new window
  // opened without saying so keeps a handle on the one it came from and tells the site it
  // arrived from where. A page holding a corpus that never left the machine says neither.
  var out = document.querySelectorAll('a[target="_blank"]');
  check('nothing on the page links back to where its numbers came from', out.length > 0);
  check('a link opens a window that can reach back into this page', Array.prototype.every.call(
    out,
    function (a) {
      var rel = a.getAttribute('rel') || '';
      return rel.indexOf('noopener') !== -1 && rel.indexOf('noreferrer') !== -1;
    }
  ));

  // Headings are how anything not reading the page top to bottom moves around it, and a
  // level skipped puts a section under something it does not belong to.
  var levels = Array.prototype.map.call(
    document.querySelectorAll('h1, h2, h3, h4, h5, h6'),
    function (h) { return Number(h.tagName.slice(1)); }
  );
  check('the page has no headings at all', levels.length > 0);
  check('the page does not start at its own title', levels[0] === 1);
  check('a heading level is skipped', levels.every(function (level, index) {
    return index === 0 || level <= levels[index - 1] + 1;
  }));

  // A control nobody can name is a control nobody can use, and the ones that open each row
  // are built by this script rather than rendered, so no Rust test ever sees them.
  var named = function (el) {
    var label = el.getAttribute('aria-label');
    if (label && label.trim().length > 0) { return true; }
    if (el.id) {
      var tag = document.querySelector('label[for="' + el.id + '"]');
      if (tag && tag.textContent.trim().length > 0) { return true; }
    }
    return el.textContent.replace(/\\s+/g, ' ').trim().length > 0;
  };
  var nameless = [];
  Array.prototype.forEach.call(
    document.querySelectorAll('button, a[href], input, summary, svg[role="img"]'),
    function (el) {
      if (!named(el)) { nameless.push(el.tagName.toLowerCase() + '.' + (el.className.baseVal || el.className || '?')); }
    }
  );
  check('something on the page can be used and not named: ' + nameless.join(', '), nameless.length === 0);

  // A bar in the chart is a bar because its height is the number it stands for, and height on
  // an SVG shape is geometry a stylesheet can overrule. A rule written for a box elsewhere in
  // the page reaches these too and flattens every month to the same few pixels, which no Rust
  // test can see: the markup is still right and only the browser knows what was drawn.
  var bars = document.querySelectorAll('figure.timeline rect.bar');
  check('the chart has no months in it', bars.length > 1);
  if (bars.length > 1) {
    var drawn = {};
    var box = null;
    Array.prototype.forEach.call(bars, function (bar) {
      box = bar.getBoundingClientRect();
      drawn[Math.round(box.height)] = true;
    });
    check('every month in the chart is drawn the same height', Object.keys(drawn).length > 1);
    var chart = document.querySelector('figure.timeline svg').getBoundingClientRect();
    check('the tallest month does not reach the top of the chart',
      Math.max.apply(null, Object.keys(drawn).map(Number)) >= chart.height * 0.9);
  }

  // A long review is clipped with a control to see the rest of it.
  var more = document.querySelector('[data-expands-text]');
  check('no long review offers the rest of itself', Boolean(more));
  if (more) {
    var text = more.previousElementSibling;
    var said = more.textContent;
    more.click();
    check('showing the rest of a review does not unclip it', text.classList.contains('open'));
    check('the control does not say the review is open',
      more.getAttribute('aria-expanded') === 'true');
    check('the control still offers what it has already done', more.textContent !== said);
    more.click();
    check('a review cannot be clipped again', !text.classList.contains('open'));
  }

  // A link from the finding to the reviews behind it.
  var link = document.querySelector('.headline a[href^="#panel-"]');
  check('the finding leads nowhere', Boolean(link));
  if (link) {
    var target = document.getElementById(link.getAttribute('href').slice(1));
    check('the finding points at nothing', Boolean(target));
    if (target) {
      // Asked for while the page is narrowed to something else, the row still has to appear.
      if (input) {
        input.value = 'zzzzzz';
        input.dispatchEvent(new Event('input'));
        check('the filter did not narrow the target away', target.classList.contains('filtered-out'));
      }
      window.location.hash = link.getAttribute('href');
      window.dispatchEvent(new HashChangeEvent('hashchange'));
      check('a link into the evidence lands on a row the filter is still hiding',
        !target.classList.contains('filtered-out'));
      check('a link into the evidence lands on a closed row', target.hidden === false);
      var opened = target.previousElementSibling.querySelector('button.disclose');
      check('a row opened by a link does not say so',
        opened && opened.getAttribute('aria-expanded') === 'true');
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

// The same page on a phone. The tables are wider than the screen there and the containers
// built to scroll them are the only thing that may reach past it, which is a property of the
// rendering and not of the markup: nothing a Rust test can see says whether it holds.
const ON_A_PHONE = `(function () {
  var wrong = [];
  var check = function (claim, ok) { if (!ok) { wrong.push(claim); } };
  var page = document.documentElement;

  check('the page itself pans sideways on a phone', page.scrollWidth <= page.clientWidth);

  var panel = document.querySelector('tr.panel');
  if (panel) {
    panel.hidden = false;
    var quoted = panel.querySelector('.text p') || panel.querySelector('p');
    if (quoted) {
      check('a quoted review is laid out wider than the phone it is read on',
        Math.round(quoted.getBoundingClientRect().width) <= page.clientWidth);
    }
    panel.hidden = true;
  }
  return wrong;
})()`;

// Every piece of text on the page, in both themes, against whatever is actually behind it.
// A palette is a set of tokens until a browser composites them, and the cells of the
// cross-game matrix are a wash of the accent colour over the card at a strength that depends
// on the number in them, which no reading of the stylesheet settles.
const CONTRAST = `(function () {
  var wrong = [];
  var root = document.documentElement;
  var chose = root.getAttribute('data-theme');

  // Everything on screen at once, whatever the checks before this left folded, filtered or
  // shut. Text nobody can see is text nobody has looked at.
  var input = document.querySelector('[data-filter-input]');
  if (input) {
    input.value = '';
    input.dispatchEvent(new Event('input'));
  }
  Array.prototype.forEach.call(document.querySelectorAll('tr.panel'), function (p) { p.hidden = false; });
  Array.prototype.forEach.call(document.querySelectorAll('details'), function (d) { d.open = true; });

  var rgba = function (value) {
    var parts = String(value).match(/[\\d.]+/g);
    if (!parts || parts.length < 3) { return null; }
    return [+parts[0], +parts[1], +parts[2], parts.length > 3 ? +parts[3] : 1];
  };
  var over = function (top, under) {
    var a = top[3];
    return [
      top[0] * a + under[0] * (1 - a),
      top[1] * a + under[1] * (1 - a),
      top[2] * a + under[2] * (1 - a),
      1
    ];
  };
  var backdrop = function (el) {
    var stack = [];
    for (var node = el; node; node = node.parentElement) {
      var colour = rgba(getComputedStyle(node).backgroundColor);
      if (colour && colour[3] > 0) {
        stack.push(colour);
        if (colour[3] === 1) { break; }
      }
    }
    var base = [255, 255, 255, 1];
    for (var i = stack.length - 1; i >= 0; i -= 1) { base = over(stack[i], base); }
    return base;
  };
  var lum = function (c) {
    var f = function (v) {
      v = v / 255;
      return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    };
    return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2]);
  };
  var ratio = function (a, b) {
    var one = lum(a);
    var two = lum(b);
    return (Math.max(one, two) + 0.05) / (Math.min(one, two) + 0.05);
  };

  var writes = function (el) {
    for (var i = 0; i < el.childNodes.length; i += 1) {
      var node = el.childNodes[i];
      if (node.nodeType === 3 && node.textContent.trim().length > 0) { return true; }
    }
    return false;
  };

  [null, 'light', 'dark'].forEach(function (theme) {
    if (theme === null) { root.removeAttribute('data-theme'); } else { root.setAttribute('data-theme', theme); }
    var worst = null;
    Array.prototype.forEach.call(document.querySelectorAll('body *'), function (el) {
      if (!writes(el)) { return; }
      var box = el.getBoundingClientRect();
      // Anything with no box on screen is folded away or read aloud rather than drawn.
      if (box.width < 4 || box.height < 4) { return; }
      var style = getComputedStyle(el);
      var ink = rgba(style.color);
      if (!ink || ink[3] === 0) { return; }
      var size = parseFloat(style.fontSize);
      var large = size >= 24 || (size >= 18.66 && parseInt(style.fontWeight, 10) >= 700);
      var found = ratio(over(ink, backdrop(el)), backdrop(el));
      if (found < (large ? 3 : 4.5) && (worst === null || found < worst.found)) {
        worst = { found: found, what: el.tagName.toLowerCase() + '.' + (el.className || ''), size: size };
      }
    });
    if (worst !== null) {
      wrong.push('text is too faint to read against what is behind it (' +
        worst.what + ', ' + worst.size + 'px, ' + worst.found.toFixed(2) + ':1, theme ' +
        (theme === null ? 'as the machine prefers' : theme) + ')');
    }
  });

  if (chose === null) { root.removeAttribute('data-theme'); } else { root.setAttribute('data-theme', chose); }
  return wrong;
})()`;

// The same page with its script taken out, which is the shape a reader with scripting off
// gets. Everything folded has to be present and open, nothing may be offered that cannot
// work, and the page must still not pan sideways.
const WITHOUT_SCRIPTING = `(function () {
  var wrong = [];
  var check = function (claim, ok) { if (!ok) { wrong.push(claim); } };
  var page = document.documentElement;

  check('the stylesheet is told scripting works when it does not', !page.classList.contains('js'));
  check('a control that only scripting can build is on the page',
    document.querySelector('button.disclose') === null);

  var form = document.querySelector('[data-filter]');
  check('a reader without scripting is offered a filter that cannot filter',
    !form || form.hidden === true);

  var pointing = document.querySelectorAll('.chevron, .arrow');
  check('a mark that says something opens is drawn where nothing opens',
    Array.prototype.every.call(pointing, function (mark) {
      return getComputedStyle(mark).display === 'none';
    }));

  var panels = document.querySelectorAll('tr.panel');
  check('there is no evidence on the page at all', panels.length > 0);
  check('evidence is folded away with nothing to unfold it',
    Array.prototype.every.call(panels, function (p) { return p.hidden === false; }));

  var clipped = document.querySelector('.text.long p');
  if (clipped) {
    check('a long review is clipped with no way to see the rest of it',
      clipped.scrollHeight <= clipped.clientHeight + 2);
  }

  check('the page itself pans sideways', page.scrollWidth <= page.clientWidth);
  return wrong;
})()`;

// Run against the same page with print media emulated, and after the reader's machine has
// been told it prefers a dark screen. Paper is white either way: a palette meant for a
// backlit panel prints as blocks of solid ink, and nothing in the markup can show it.
const ON_PAPER = `(function () {
  var wrong = [];
  var lit = function (colour) {
    var parts = String(colour).match(/[\\d.]+/g);
    if (!parts || parts.length < 3) { return 0; }
    if (parts.length > 3 && Number(parts[3]) === 0) { return 255; }
    return (Number(parts[0]) + Number(parts[1]) + Number(parts[2])) / 3;
  };
  var paper = function (selector) {
    var el = document.querySelector(selector);
    return el ? lit(getComputedStyle(el).backgroundColor) : 255;
  };
  var root = document.documentElement;
  var chose = root.getAttribute('data-theme');
  [null, 'dark'].forEach(function (theme) {
    if (theme === null) { root.removeAttribute('data-theme'); } else { root.setAttribute('data-theme', theme); }
    var who = theme === null ? 'the machine prefers one' : 'the reader asked for one';
    if (lit(getComputedStyle(document.body).color) > 128) {
      wrong.push('the printed page is written in light ink where ' + who);
    }
    if (paper('.headline') < 200) {
      wrong.push('a finding prints as a dark block where ' + who);
    }
    if (paper('nav.contents, section.game') < 200) {
      wrong.push('a section prints as a dark block where ' + who);
    }
  });
  if (chose === null) { root.removeAttribute('data-theme'); } else { root.setAttribute('data-theme', chose); }
  return wrong;
})()`;

const file = resolve(process.argv[2] ?? "sample-report.html");
const page = pathToFileURL(file).href;
const profile = await mkdtemp(join(tmpdir(), "census-report-check-"));

// The same page with the script cut out, which is exactly what a reader with scripting off
// is served. Driven as a page of its own rather than by turning scripting off in the
// browser, because a debugger that cannot run script cannot ask the page anything either.
const mute = join(profile, "without-scripting.html");
await writeFile(
  mute,
  (await readFile(file, "utf8")).replace(/<script>[\s\S]*?<\/script>/g, ""),
);
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
  // Loaded once already by the launch, before anything could watch it, so it is loaded again
  // with the network being listened to.
  await send("Network.enable", {});
  await send("Page.reload", { ignoreCache: true });
  for (let attempt = 0; attempt < 80; attempt += 1) {
    const ready = await evaluate(
      "document.readyState === 'complete' && document.documentElement.classList.contains('js')",
    );
    if (ready.result?.result?.value === true) break;
    await sleep(250);
  }

  const fetched = asked.filter((url) => url !== page);
  const answer = await evaluate(PROBE);
  await send("Emulation.setDeviceMetricsOverride", {
    width: 420,
    height: 900,
    deviceScaleFactor: 1,
    mobile: false,
  });
  const phone = await evaluate(ON_A_PHONE);
  await send("Emulation.clearDeviceMetricsOverride", {});
  const ink = await evaluate(CONTRAST);
  await send("Emulation.setEmulatedMedia", {
    media: "print",
    features: [{ name: "prefers-color-scheme", value: "dark" }],
  });
  const paper = await evaluate(ON_PAPER);

  await send("Emulation.setEmulatedMedia", { media: "screen" });
  await send("Page.navigate", { url: pathToFileURL(mute).href });
  for (let attempt = 0; attempt < 80; attempt += 1) {
    const ready = await evaluate(
      "document.readyState === 'complete' && document.querySelectorAll('tr.panel').length > 0",
    );
    if (ready.result?.result?.value === true) break;
    await sleep(250);
  }
  const quiet = await evaluate(WITHOUT_SCRIPTING);

  socket.close();
  const threw = [answer, phone, ink, paper, quiet].find((r) => r.result?.exceptionDetails);
  if (threw) {
    console.error("the page threw while being checked:");
    console.error(threw.result.exceptionDetails.exception?.description ?? threw.result.exceptionDetails.text);
  } else {
    const wrong = (fetched.length === 0
      ? []
      : [`the page fetched ${fetched.length} thing(s): ${fetched.slice(0, 5).join(", ")}`]
    )
      .concat(answer.result.result.value)
      .concat(phone.result.result.value)
      .concat(ink.result.result.value)
      .concat(paper.result.result.value)
      .concat(quiet.result.result.value.map((claim) => `${claim}, with scripting off`));
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
