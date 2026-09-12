const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const whole = new Intl.NumberFormat();
const share = new Intl.NumberFormat(undefined, { style: 'percent', maximumFractionDigits: 2 });

const el = (id) => document.getElementById(id);
const shelf = el('shelf');
const views = {
  welcome: el('welcome'),
  finder: el('finder'),
  game: el('game'),
  evidence: el('evidence'),
};

const PER_PAGE = 25;
const day = new Intl.DateTimeFormat(undefined, { year: 'numeric', month: 'short', day: 'numeric' });

let games = [];
let chosen = null;
let busy = false;
/* The language the shown reading counts, or null for every language. */
let readLanguage = null;

function show(which) {
  for (const [name, node] of Object.entries(views)) node.hidden = name !== which;
}

function set(node, text) {
  node.textContent = text;
}

/* Reviews come from Valve and their titles come with them, so nothing here is ever built by
   parsing text into markup. */
function facts(list, entries) {
  list.replaceChildren();
  for (const [term, value, under] of entries) {
    const wrap = document.createElement('div');
    const dt = document.createElement('dt');
    dt.textContent = term;
    const dd = document.createElement('dd');
    dd.textContent = value;
    if (under) {
      const small = document.createElement('small');
      small.textContent = under;
      dd.append(small);
    }
    wrap.append(dt, dd);
    list.append(wrap);
  }
}

function drawShelf() {
  const needle = el('filter').value.trim().toLowerCase();
  const shown = needle
    ? games.filter((game) => game.name.toLowerCase().includes(needle) || String(game.app_id).includes(needle))
    : games;

  shelf.replaceChildren();
  for (const game of shown) {
    const item = document.createElement('li');
    const button = document.createElement('button');
    button.type = 'button';
    if (game.app_id === chosen) button.setAttribute('aria-current', 'true');

    const title = document.createElement('span');
    title.className = 'title';
    title.textContent = game.name;

    const meta = document.createElement('span');
    meta.className = 'meta';
    const pip = document.createElement('span');
    pip.className = `pip ${game.stage}`;
    const count = document.createElement('span');
    count.textContent = `${whole.format(game.reviews)} reviews`;
    meta.append(pip, count);

    button.append(title, meta);
    button.addEventListener('click', () => choose(game.app_id));
    item.append(button);
    shelf.append(item);
  }
}

function choose(appId) {
  chosen = appId;
  const game = games.find((one) => one.app_id === appId);
  if (!game) return;
  drawShelf();
  set(el('game-name'), game.name);
  set(el('game-sub'), game.verdict ? `${game.verdict} on Steam` : `App ${game.app_id}`);
  facts(el('game-facts'), [
    ['Held here', whole.format(game.reviews), 'reviews downloaded'],
    ['Valve reports', whole.format(game.valve_total), 'reviews in total'],
    ['Coverage', share.format(game.coverage), 'of what Valve serves'],
    ['Stage', stageName(game.stage), null],
  ]);
  set(el('game-note'), '');
  el('game-note').classList.remove('bad');
  el('work').hidden = true;
  el('sweep-actions').hidden = busy;
  show('game');
  loadTopics(game);
}

/* Fetches what was written or edited since the capture was made, then re-reads the game if
   it had been read, because counts over a corpus that has since changed are counts over a
   corpus nobody can open. */
async function sweepGame() {
  if (busy || chosen === null) return;
  busy = true;
  const appId = chosen;
  el('game-actions').hidden = true;
  el('sweep-actions').hidden = true;
  el('work').hidden = false;
  set(el('work-what'), 'Asking Steam what changed');
  set(el('work-count'), '');
  el('work-fill').style.width = '100%';
  el('work-fill').classList.add('working');
  set(el('game-note'), '');

  try {
    const swept = await invoke('sweep', { appId });
    busy = false;
    await refresh();
    if (swept.rows > 0 && !el('topics').hidden) await readGame(readLanguage);
    if (appId === chosen) {
      set(
        el('game-note'),
        swept.rows === 0
          ? `Nothing was written or edited since ${day.format(new Date(swept.since * 1000))}.`
          : `${whole.format(swept.rows)} reviews fetched: ${whole.format(swept.new)} new and ` +
              `${whole.format(swept.edited)} edited since ${day.format(new Date(swept.since * 1000))}.`,
      );
    }
  } catch (failure) {
    const note = el('game-note');
    note.classList.add('bad');
    set(note, String(failure));
  } finally {
    busy = false;
    el('work').hidden = true;
    el('sweep-actions').hidden = false;
  }
}

listen('sweep', ({ payload }) => {
  if (payload.app_id !== chosen) return;
  set(el('work-what'), `Page ${whole.format(payload.pages)}`);
  set(el('work-count'), `${whole.format(payload.rows)} changed`);
});

async function loadTopics(game) {
  const panel = el('topics');
  try {
    const counted = await invoke('reading', { appId: game.app_id });
    if (counted.app_id !== chosen) return;
    drawTopics(counted);
    drawTimeline(counted.months);
    drawLanguages(counted.languages, counted.corpus_reviews);
    panel.hidden = false;
    el('game-actions').hidden = true;
    set(el('game-note'), '');
    loadInduced(game.app_id);
  } catch (failure) {
    panel.hidden = true;
    el('game-actions').hidden = busy;
    set(
      el('game-note'),
      String(failure).includes('not been read')
        ? 'Downloaded but not read yet. Reading turns it into rates you can open.'
        : String(failure),
    );
  }
}

/* `language` is a Steam language name to count only, or null for every language. Left
   unset, it is whatever the control beside the button says, which is how a first reading
   chooses; a reading that exists already is re-read in the language it was made in, or in
   the other one when the reader asks to switch. */
async function readGame(language = undefined) {
  if (busy || chosen === null) return;
  busy = true;
  const appId = chosen;
  const wanted = language === undefined ? el('read-language').value || null : language;
  el('game-actions').hidden = true;
  el('sweep-actions').hidden = true;
  el('work').hidden = false;
  set(el('work-what'), 'Starting');
  set(el('work-count'), '');
  el('work-fill').style.width = '0%';
  set(el('game-note'), '');

  try {
    await invoke('read_game', { appId, language: wanted });
    await refresh();
  } catch (failure) {
    const note = el('game-note');
    note.classList.add('bad');
    set(note, String(failure));
    el('game-actions').hidden = false;
  } finally {
    busy = false;
    el('work').hidden = true;
    el('sweep-actions').hidden = false;
  }
}

listen('fetch', ({ payload }) => {
  if (!busy) return;
  const mb = (bytes) => `${Math.round(bytes / 1e6)} MB`;
  set(el('work-what'), `Fetching the model, ${payload.file}`);
  set(
    el('work-count'),
    payload.total === null ? mb(payload.downloaded) : `${mb(payload.downloaded)} of ${mb(payload.total)}`,
  );
  /* The one bar in the window that does have a denominator: the server said how big the file
     is, so the fill can mean something. */
  el('work-fill').classList.remove('working');
  el('work-fill').style.width =
    payload.total === null ? '100%' : `${(100 * payload.downloaded) / payload.total}%`;
});

listen('read', ({ payload }) => {
  if (payload.app_id !== chosen) return;
  set(el('work-what'), 'Reading each point');
  set(
    el('work-count'),
    `${whole.format(payload.claims_read)} read, ` +
      `${whole.format(payload.reviews_counted)} reviews counted`,
  );
  /* No total to divide by: how many distinct points a corpus holds is not known until it has
     been walked, and a bar that invents a denominator is a bar that lies. */
  el('work-fill').style.width = '100%';
  el('work-fill').classList.add('working');
});

function cell(text, className) {
  const td = document.createElement('td');
  td.className = className ? `num ${className}` : 'num';
  const span = document.createElement('span');
  span.textContent = text;
  td.append(span);
  return td;
}

const SVG = 'http://www.w3.org/2000/svg';
const CHART = { width: 1000, height: 160 };

function shape(name, attributes) {
  const node = document.createElementNS(SVG, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  return node;
}

/* Reviews per month as bars against the busiest month, and the share recommending the game
   as a line from none at the bottom to all at the top. The same two shapes the report page
   draws, in the same coordinate space, so a reader moving between them sees one chart. A
   rate is drawn only for a month big enough to carry one; the core decides which those are. */
function drawTimeline(months) {
  const figure = el('timeline');
  figure.hidden = months.length < 2;
  if (figure.hidden) return;

  const peak = months.reduce((best, month) => (month.reviews > best.reviews ? month : best));
  const tallest = Math.max(peak.reviews, 1);
  const step = CHART.width / months.length;
  const svg = el('timeline-svg');
  svg.replaceChildren();
  svg.setAttribute('aria-label', `Reviews per month from ${months[0].name} to ${months[months.length - 1].name}`);

  months.forEach((month, index) => {
    const tall = (month.reviews / tallest) * CHART.height;
    svg.append(
      shape('rect', {
        class: 'bar',
        x: (index * step).toFixed(2),
        y: (CHART.height - tall).toFixed(2),
        width: Math.max(step * 0.82, 0.5).toFixed(2),
        height: tall.toFixed(2),
      }),
    );
  });
  svg.append(
    shape('line', {
      class: 'midline',
      x1: 0,
      y1: (CHART.height / 2).toFixed(2),
      x2: CHART.width,
      y2: (CHART.height / 2).toFixed(2),
    }),
  );
  const points = months
    .map((month, index) =>
      month.positive === null
        ? null
        : `${(index * step + step / 2).toFixed(2)},${((1 - month.positive) * CHART.height).toFixed(2)}`,
    )
    .filter((point) => point !== null);
  if (points.length > 1) svg.append(shape('polyline', { class: 'share', points: points.join(' ') }));

  /* Full-height hit areas last, because a quiet month is a bar one pixel tall and nothing
     to point at. The title is the only way a month is read. */
  months.forEach((month, index) => {
    const hit = shape('rect', {
      class: 'hit',
      x: (index * step).toFixed(2),
      y: 0,
      width: Math.max(step, 0.5).toFixed(2),
      height: CHART.height,
    });
    const title = document.createElementNS(SVG, 'title');
    title.textContent = `${month.name}: ${whole.format(month.reviews)} reviews, ${
      month.positive === null ? 'too few to carry a rate' : `${share.format(month.positive)} recommended`
    }`;
    hit.append(title);
    svg.append(hit);
  });

  set(
    el('timeline-note'),
    `Reviews per month, against a busiest month of ${whole.format(tallest)} in ${peak.name}. ` +
      `The line is the share of each month that recommended the game; the dashed line is half. ` +
      `Point at a month to read it.`,
  );
  set(el('timeline-first'), months[0].name);
  set(el('timeline-last'), months[months.length - 1].name);
}

/* The languages of the whole capture, commonest first, so a reader knows what "reviews"
   means before reading a rate over them. */
function drawLanguages(languages, corpus) {
  const line = el('languages');
  line.hidden = languages.length === 0;
  if (line.hidden) return;
  const shown = languages.slice(0, 8);
  const named = shown.map((language) =>
    language.share === null
      ? language.name
      : `${language.name} ${share.format(language.share)}`,
  );
  const rest = languages.length - shown.length;
  set(
    line,
    `Languages in the ${whole.format(corpus)} captured reviews: ${named.join(', ')}` +
      (rest > 0 ? `, and ${whole.format(rest)} more.` : '.'),
  );
}

/* Loaded after the table rather than with it: it walks the capture for the quoted reviews,
   and most games have not had their own subjects induced yet, so most of the time there is
   nothing to show and nothing to wait for. */
async function loadInduced(appId) {
  const section = el('induced');
  section.hidden = true;
  let found;
  try {
    found = await invoke('induced', { appId });
  } catch {
    return;
  }
  if (appId !== chosen || found.length === 0) return;

  const list = el('induced-list');
  list.replaceChildren();
  for (const subject of found) {
    const wrap = document.createElement('div');
    wrap.className = 'found';
    const dt = document.createElement('dt');
    dt.textContent = subject.label;
    if (subject.refines !== null) {
      const form = document.createElement('span');
      form.className = 'quiet';
      form.textContent = ` a form of ${subject.refines}`;
      dt.append(form);
    }
    const dd = document.createElement('dd');
    const about = document.createElement('p');
    about.textContent = subject.description;
    const details = document.createElement('details');
    const summary = document.createElement('summary');
    summary.textContent = `${whole.format(subject.reviews.length)} of the ${whole.format(
      subject.found_in,
    )} reviews it was found in`;
    details.append(summary);
    const quotes = document.createElement('ol');
    quotes.className = 'quotes';
    for (const review of subject.reviews) quotes.append(quoteItem(review));
    details.append(quotes);
    dd.append(about, details);
    wrap.append(dt, dd);
    list.append(wrap);
  }
  section.hidden = false;
}

/* A review shown whole, with the facts about it and the way back to Steam, and no reading:
   the model that counts the table never read it, so a confidence here would be invented. */
function quoteItem(review) {
  const item = document.createElement('li');
  const body = document.createElement('p');
  body.lang = bcp47(review.language);
  body.textContent = review.review;
  const byline = document.createElement('div');
  byline.className = 'byline';
  const verdict = document.createElement('span');
  verdict.className = review.voted_up ? 'verdict-up' : 'verdict-down';
  verdict.textContent = review.voted_up ? 'Recommended the game' : 'Did not recommend it';
  byline.append(verdict);
  if (review.votes_up > 0) {
    const votes = document.createElement('span');
    votes.textContent = `${whole.format(review.votes_up)} found it helpful`;
    byline.append(votes);
  }
  const when = document.createElement('span');
  when.textContent = day.format(new Date(review.created * 1000));
  byline.append(when);
  if (review.url) {
    const link = document.createElement('a');
    link.href = '#';
    link.textContent = 'On Steam';
    link.addEventListener('click', (event) => {
      event.preventDefault();
      openOutside(review.url);
    });
    byline.append(link);
  }
  item.append(body, byline);
  return item;
}

function drawTopics(counted) {
  const ranked = counted.subjects
    .filter((subject) => subject.reviews > 0)
    .sort((left, right) => right.reviews - left.reviews);
  const widest = ranked.length > 0 ? (ranked[0].rate ?? 0) : 0;

  const parts = [`${whole.format(counted.reviews)} reviews`];
  if (counted.language) {
    parts[0] = `${whole.format(counted.reviews)} ${counted.language} reviews of ${whole.format(
      counted.corpus_reviews,
    )} in the corpus`;
  }
  parts.push(`${whole.format(counted.claims)} separate points`);
  if (counted.positive_baseline !== null) {
    parts.push(`${share.format(counted.positive_baseline)} recommending the game`);
  }
  set(el('topics-counted'), `${parts.join(', ')}. `);

  /* The other reading is one click away, and it is a re-read rather than a filter: the
     model has to read the claims it skipped. */
  readLanguage = counted.language;
  set(
    el('switch-language'),
    counted.language ? 'Count every language instead' : 'Count only English reviews instead',
  );

  /* The paragraph is assembled by the core from the same counts the table shows, so the
     window only decides whether there is one to show. */
  const summary = el('in-short');
  summary.hidden = counted.in_short === '';
  set(summary, counted.in_short);

  const swept = el('swept-caveat');
  swept.hidden = counted.swept_since === null;
  if (counted.swept_since !== null) {
    set(
      swept,
      `The capture was brought up to date on ${day.format(new Date(counted.swept_since * 1000))} ` +
        `and these counts were made before that. Read it again to count what arrived.`,
    );
  }

  /* The counts stand whatever cut them; a claim quoted by its index does not, because this
     build would cut the review into different pieces. */
  const cut = el('split-caveat');
  cut.hidden = !counted.older_splitter;
  set(
    cut,
    `These counts were made with an older way of taking reviews apart, so the points behind ` +
      `them cannot be shown until the game is read again.`,
  );

  const unread = counted.claims > 0 ? counted.unclassified_claims / counted.claims : 0;
  const silent = counted.reviews > 0 ? counted.silent_reviews / counted.reviews : 0;

  /* Above the table, not below it, once the model has declined more than it answered. Every
     rate under it is then a floor, and a reader who finds that out in a footnote has already
     drawn the wrong conclusion. */
  const caveat = el('topics-caveat');
  caveat.hidden = unread < 0.5;
  set(
    caveat,
    `The model would not commit to a subject for ${share.format(unread)} of the points ` +
      `here, and said nothing at all about ${share.format(silent)} of the reviews. Those are ` +
      `counted as unclassified rather than filed under whatever came closest, so every rate ` +
      `below is a floor: what it was sure of, not everything that was said.`,
  );

  /* How often it is wrong, where that has been measured. A rate without this beside it is an
     opinion with decimal places, and a game nobody has labelled is told so rather than
     shown the same table with nothing missing from it. */
  const measured = counted.measured;
  const frozen = counted.frozen;
  const trust =
    measured === null
      ? frozen
        ? /* Most games anyone opens are in exactly this position, and saying only that
             nothing is measured invites the reader to distrust everything or to trust
             everything. The model does have a measurement; it is about other games, and the
             wording has to say so. */
          `Nobody has labelled this game's claims, so how often the model is wrong here has ` +
          `not been measured. What is measured is ${frozen.games} games it had never seen, ` +
          `over ${whole.format(frozen.claims)} labelled claims: it answers ` +
          `${share.format(frozen.coverage)} of them and names the same subject a separate ` +
          `labeller did ${share.format(frozen.accuracy)} of the time when it does. Expect ` +
          `this game to be near that, and treat every rate as provisional until it is ` +
          `labelled too.`
        : `Nobody has labelled this game's claims, so how often the model is wrong here has ` +
          `not been measured. Treat every rate as provisional.`
      : measured.agreement === null
        ? `The model declined every labelled claim on this game, so nothing here is measured.`
        : `Where this game has been labelled, the model named the same subject a separate ` +
          `labeller did ${share.format(measured.agreement)} of the time on the ` +
          `${whole.format(measured.answered)} claims it answered, somewhere between ` +
          `${share.format(measured.low)} and ${share.format(measured.high)}. That is agreement ` +
          `with another model, not accuracy.` +
          /* A label names a span of a review, and a splitter that has since learned to cut
             that review differently leaves the label naming nothing. */
          (measured.unjoined > 0
            ? ` A further ${whole.format(measured.unjoined)} labelled claims are left out ` +
              `because this build takes their reviews apart differently from the build they ` +
              `were labelled under.`
            : ``);

  set(
    el('topics-footnote'),
    `Mention rates count a review once for every subject it raises, however many times it ` +
      `raises it, so they add up to more than 100% and are meant to. ` +
      (caveat.hidden
        ? `${share.format(unread)} of points name no subject the model would commit to, and ` +
          `${whole.format(counted.silent_reviews)} reviews name none at all. Those are ` +
          `counted here rather than filed under whatever came closest. `
        : `Every row opens onto the points behind it. `) +
      trust,
  );

  const rows = el('topic-rows');
  rows.replaceChildren();
  for (const subject of ranked) {
    const row = document.createElement('tr');

    const name = document.createElement('td');
    const open = document.createElement('button');
    open.type = 'button';
    open.className = 'subject';
    open.textContent = subject.label;
    open.addEventListener('click', () => openClaims(subject, 0));
    name.append(open);
    /* Marked where the model is measured to miss most of a subject's labelled claims: that
       row's rate is a floor, and a reader scanning the table cannot tell it from a count
       unless the row says so. */
    if (subject.found !== null && subject.found < 0.25) {
      const thin = document.createElement('span');
      thin.className = 'thin';
      thin.title = `Found in only ${share.format(subject.found)} of the labelled claims about it, so this rate is a floor rather than a count`;
      thin.textContent = '!';
      name.append(thin);
    }
    /* The corrected share, where the measured errors allow one. Shown as a hint on the name
       rather than a column of its own, because most games have no labels and a column that
       is empty for most of them teaches a reader to skip it. */
    if (subject.corrected !== null) {
      const fixed = document.createElement('span');
      fixed.className = 'corrected';
      fixed.title = 'The share of points about this with the model’s measured errors taken out';
      fixed.textContent = `≈ ${share.format(subject.corrected)} of points`;
      name.append(fixed);
    }

    const rate = document.createElement('td');
    rate.className = 'num rate';
    const value = document.createElement('span');
    value.textContent = subject.rate === null ? '—' : share.format(subject.rate);
    const bar = document.createElement('i');
    bar.className = 'bar';
    bar.style.transform = `scaleX(${widest > 0 ? (subject.rate ?? 0) / widest : 0})`;
    rate.append(value, bar);

    const gauge = document.createElement('td');
    gauge.className = 'num';
    const factor = document.createElement('span');
    if (subject.bias === null) {
      factor.textContent = '—';
      factor.className = 'faint';
    } else {
      factor.textContent = `${subject.bias.toFixed(1)}×`;
      factor.className = subject.bias >= 1.15 ? 'over' : subject.bias <= 0.87 ? 'under' : 'faint';
    }
    gauge.append(factor);

    row.append(
      name,
      rate,
      cell(whole.format(subject.praised), 'under'),
      cell(whole.format(subject.criticised), 'over'),
      cell(whole.format(subject.mixed), 'faint'),
      gauge,
    );
    rows.append(row);
  }
}

/* `narrowed` is null for every point under the subject, or `{side, term}` for the points on
   one side that use a word from the strip. The same page function serves both, so the strip
   is a filter on the evidence and not a second view of it. */
async function openClaims(subject, from, narrowed = null) {
  reading = { subject, from, narrowed };
  show('evidence');
  set(el('evidence-name'), subject.label);
  set(el('evidence-lede'), 'Finding them...');
  drawTerms(subject, narrowed);
  el('quotes').replaceChildren();
  el('earlier').disabled = true;
  el('later').disabled = true;

  let page;
  try {
    page = await invoke('claims_behind', {
      appId: chosen,
      subject: subject.id,
      side: narrowed?.side ?? null,
      term: narrowed?.term ?? null,
      from,
      count: PER_PAGE,
    });
  } catch (failure) {
    set(el('evidence-lede'), String(failure));
    return;
  }
  if (reading?.subject.id !== subject.id || reading.from !== from || reading.narrowed !== narrowed) {
    return;
  }

  set(
    el('evidence-lede'),
    narrowed === null
      ? `${whole.format(page.total)} separate points about this, raised in ` +
          `${whole.format(subject.reviews)} reviews. Each one is shown as it was written.`
      : `${whole.format(page.total)} ${narrowed.side === 'praise' ? 'praising' : 'complaining'} ` +
          `points about this that say “${narrowed.term}”. Each one is shown as it was written.`,
  );
  drawClaims(page.claims);

  const upTo = from + page.claims.length;
  set(el('paging-note'), `${whole.format(from + 1)} to ${whole.format(upTo)}`);
  el('earlier').disabled = from === 0;
  el('later').disabled = upTo >= page.total;
}

/* The words each side of a subject uses and the other does not, each a button that narrows
   the page to the points using it. Nothing is drawn for a side with nothing to say, and the
   whole strip goes when neither side has anything, which on a game the model barely reads is
   the honest state rather than an empty box. */
function drawTerms(subject, narrowed) {
  const sides = [
    ['praised-terms', 'praise', subject.praised_terms],
    ['criticised-terms', 'complaint', subject.criticised_terms],
  ];
  let any = false;
  for (const [id, side, terms] of sides) {
    const strip = el(id);
    strip.hidden = terms.length === 0;
    for (const stale of strip.querySelectorAll('button')) stale.remove();
    for (const term of terms) {
      any = true;
      const chip = document.createElement('button');
      chip.type = 'button';
      const chosenHere = narrowed !== null && narrowed.side === side && narrowed.term === term.text;
      chip.className = chosenHere ? 'term chosen' : 'term';
      chip.title = chosenHere
        ? 'Back to every point about this'
        : `${whole.format(term.reviews)} reviews used this word on this side`;
      chip.append(term.text);
      const count = document.createElement('span');
      count.className = 'n';
      count.textContent = whole.format(term.reviews);
      chip.append(count);
      chip.addEventListener('click', () =>
        openClaims(subject, 0, chosenHere ? null : { side, term: term.text }),
      );
      strip.append(chip);
    }
  }
  el('stands-out').hidden = !any;
}

function drawClaims(claims) {
  const list = el('quotes');
  list.replaceChildren();
  for (const found of claims) {
    const item = document.createElement('li');

    const body = document.createElement('p');
    body.lang = bcp47(found.language);
    /* The claim is shown inside the review it came from, so a reader can see whether it was
       cut in the right place rather than taking the split on trust. */
    const at = found.review.indexOf(found.claim);
    if (at === -1) {
      body.textContent = found.claim;
    } else {
      const before = document.createElement('span');
      before.className = 'quiet';
      before.textContent = found.review.slice(Math.max(0, at - 160), at);
      const it = document.createElement('b');
      it.textContent = found.claim;
      const after = document.createElement('span');
      after.className = 'quiet';
      after.textContent = found.review.slice(at + found.claim.length, at + found.claim.length + 160);
      body.append(before, it, after);
    }

    const byline = document.createElement('div');
    byline.className = 'byline';

    const polarity = document.createElement('span');
    polarity.className =
      found.polarity === 'praise' ? 'verdict-up' : found.polarity === 'complaint' ? 'verdict-down' : '';
    polarity.textContent =
      found.polarity === 'praise' ? 'Praise' : found.polarity === 'complaint' ? 'Complaint' : 'Neutral';
    byline.append(polarity);

    const sure = document.createElement('span');
    sure.textContent = `${share.format(found.confidence)} sure`;
    byline.append(sure);

    const verdict = document.createElement('span');
    verdict.textContent = found.voted_up ? 'Recommended the game' : 'Did not recommend it';
    byline.append(verdict);

    if (found.votes_up > 0) {
      const votes = document.createElement('span');
      votes.textContent = `${whole.format(found.votes_up)} found it helpful`;
      byline.append(votes);
    }

    const when = document.createElement('span');
    when.textContent = day.format(new Date(found.created * 1000));
    byline.append(when);

    if (found.url) {
      const link = document.createElement('a');
      link.href = '#';
      link.textContent = 'On Steam';
      link.addEventListener('click', (event) => {
        event.preventDefault();
        openOutside(found.url);
      });
      byline.append(link);
    }

    item.append(body, byline);
    list.append(item);
  }
}

let reading = null;

/* Steam's language codes are its own; the ones a browser needs for hyphenation and font
   selection are not, and getting this wrong renders Chinese in a Japanese face. */
function bcp47(steam) {
  const known = {
    schinese: 'zh-Hans',
    tchinese: 'zh-Hant',
    japanese: 'ja',
    koreana: 'ko',
    russian: 'ru',
    thai: 'th',
    brazilian: 'pt-BR',
    latam: 'es-419',
    english: 'en',
    french: 'fr',
    german: 'de',
    spanish: 'es',
    italian: 'it',
    polish: 'pl',
    turkish: 'tr',
    ukrainian: 'uk',
    czech: 'cs',
    dutch: 'nl',
    hungarian: 'hu',
    portuguese: 'pt',
    swedish: 'sv',
    danish: 'da',
    finnish: 'fi',
    norwegian: 'no',
    romanian: 'ro',
    bulgarian: 'bg',
    greek: 'el',
    vietnamese: 'vi',
    indonesian: 'id',
  };
  return known[steam] ?? '';
}

function openOutside(url) {
  const opener = window.__TAURI__.opener;
  if (opener?.openUrl) opener.openUrl(url);
  else invoke('plugin:opener|open_url', { url });
}

function stageName(stage) {
  if (stage === 'read') return 'Read';
  if (stage === 'embedded') return 'Embedded';
  return 'Downloaded';
}

async function refresh() {
  const held = await invoke('library');
  games = held.games;
  set(el('where'), held.path);
  drawShelf();
  if (chosen !== null && games.some((game) => game.app_id === chosen)) choose(chosen);
  else if (games.length > 0) choose(games[0].app_id);
  else show('welcome');
}

function openFinder() {
  show('finder');
  el('found').hidden = true;
  set(el('lookup-note'), '');
  el('lookup-note').classList.remove('bad');
  el('appid').value = '';
  el('appid').focus();
}

let found = null;

async function lookUp(event) {
  event.preventDefault();
  const appId = Number.parseInt(el('appid').value.trim(), 10);
  const note = el('lookup-note');
  note.classList.remove('bad');
  if (!Number.isInteger(appId) || appId <= 0) {
    note.classList.add('bad');
    set(note, 'An app ID is the number in the store URL, digits only.');
    return;
  }

  el('lookup').disabled = true;
  set(note, 'Asking Steam...');
  try {
    found = await invoke('look_up', { appId });
    set(note, '');
    set(el('found-name'), found.name || `App ${found.app_id}`);
    facts(el('found-facts'), [
      ['Reviews', whole.format(found.reviews), 'Valve will serve'],
      ['Positive', whole.format(found.positive), null],
      ['Negative', whole.format(found.negative), null],
      ['Verdict', found.verdict || 'None yet', null],
    ]);
    set(
      el('found-note'),
      found.held
        ? 'Already in your library. Downloading again picks up where the last crawl stopped.'
        : `About ${whole.format(Math.ceil(found.reviews / 100))} requests, paced so Valve is not leaned on.`,
    );
    el('found').hidden = false;
  } catch (failure) {
    note.classList.add('bad');
    set(note, String(failure));
  } finally {
    el('lookup').disabled = false;
  }
}

async function start() {
  if (busy || !found) return;
  busy = true;
  const appId = found.app_id;
  const name = found.name || `App ${appId}`;

  chosen = appId;
  show('game');
  set(el('game-name'), name);
  set(el('game-sub'), 'Downloading every review');
  facts(el('game-facts'), []);
  set(el('game-note'), '');
  el('work').hidden = false;
  set(el('work-what'), 'Starting');
  set(el('work-count'), '');
  el('work-fill').style.width = '0%';

  try {
    const held = await invoke('crawl', { appId });
    games = held.games;
    set(el('where'), held.path);
    drawShelf();
    choose(appId);
  } catch (failure) {
    el('work').hidden = true;
    const note = el('game-note');
    note.classList.add('bad');
    set(note, String(failure));
  } finally {
    busy = false;
  }
}

listen('crawl', ({ payload }) => {
  if (payload.app_id !== chosen) return;
  const done = payload.shards_total ? payload.shards_done / payload.shards_total : 0;
  el('work-fill').style.width = `${(done * 100).toFixed(1)}%`;
  set(el('work-what'), `Downloading, ${payload.shards_done} of ${payload.shards_total} windows`);
  set(
    el('work-count'),
    payload.valve_total
      ? `${whole.format(payload.unique)} of ${whole.format(payload.valve_total)}`
      : whole.format(payload.unique),
  );
});

el('filter').addEventListener('input', drawShelf);
el('add').addEventListener('click', openFinder);
el('welcome-add').addEventListener('click', openFinder);
el('lookup-form').addEventListener('submit', lookUp);
el('start').addEventListener('click', start);
el('cancel').addEventListener('click', () => (chosen === null ? show('welcome') : choose(chosen)));
el('back').addEventListener('click', () => choose(chosen));
el('do-read').addEventListener('click', () => readGame());
el('do-sweep').addEventListener('click', sweepGame);
el('switch-language').addEventListener('click', () => readGame(readLanguage ? null : 'english'));
el('earlier').addEventListener('click', () => {
  if (reading) openClaims(reading.subject, Math.max(0, reading.from - PER_PAGE), reading.narrowed);
});
el('later').addEventListener('click', () => {
  if (reading) openClaims(reading.subject, reading.from + PER_PAGE, reading.narrowed);
});

refresh();
