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
  el('work').hidden = true;
  show('game');
  loadTopics(game);
}

async function loadTopics(game) {
  const panel = el('topics');
  try {
    const counted = await invoke('reading', { appId: game.app_id });
    if (counted.app_id !== chosen) return;
    drawTopics(counted);
    panel.hidden = false;
    el('game-actions').hidden = true;
    set(el('game-note'), '');
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

async function readGame() {
  if (busy || chosen === null) return;
  busy = true;
  const appId = chosen;
  el('game-actions').hidden = true;
  el('work').hidden = false;
  set(el('work-what'), 'Starting');
  set(el('work-count'), '');
  el('work-fill').style.width = '0%';
  set(el('game-note'), '');

  try {
    await invoke('read_game', { appId, language: 'english' });
    await refresh();
  } catch (failure) {
    const note = el('game-note');
    note.classList.add('bad');
    set(note, String(failure));
    el('game-actions').hidden = false;
  } finally {
    busy = false;
    el('work').hidden = true;
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
  set(
    el('work-what'),
    payload.reading_claims ? 'Reading each point' : 'Counting what they add up to',
  );
  set(el('work-count'), whole.format(payload.done));
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
  set(el('topics-lede'), `${parts.join(', ')}.`);

  const unread = counted.claims > 0 ? counted.unclassified_claims / counted.claims : 0;
  set(
    el('topics-footnote'),
    `Mention rates count a review once for every subject it raises, however many times it ` +
      `raises it, so they add up to more than 100% and are meant to. ` +
      `${share.format(unread)} of points name no subject the model would commit to, and ` +
      `${whole.format(counted.silent_reviews)} reviews name none at all. Those are counted ` +
      `here rather than filed under whatever came closest.`,
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

async function openClaims(subject, from) {
  reading = { subject, from };
  show('evidence');
  set(el('evidence-name'), subject.label);
  set(el('evidence-lede'), 'Finding them...');
  el('quotes').replaceChildren();
  el('earlier').disabled = true;
  el('later').disabled = true;

  let page;
  try {
    page = await invoke('claims_behind', {
      appId: chosen,
      subject: subject.id,
      from,
      count: PER_PAGE,
    });
  } catch (failure) {
    set(el('evidence-lede'), String(failure));
    return;
  }
  if (reading?.subject.id !== subject.id || reading.from !== from) return;

  set(
    el('evidence-lede'),
    `${whole.format(page.total)} separate points about this, raised in ` +
      `${whole.format(subject.reviews)} reviews. Each one is shown as it was written.`,
  );
  drawClaims(page.claims);

  const upTo = from + page.claims.length;
  set(el('paging-note'), `${whole.format(from + 1)} to ${whole.format(upTo)}`);
  el('earlier').disabled = from === 0;
  el('later').disabled = upTo >= page.total;
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
  if (stage === 'classified') return 'Counted';
  if (stage === 'embedded') return 'Read';
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
el('do-read').addEventListener('click', readGame);
el('earlier').addEventListener('click', () => {
  if (reading) openClaims(reading.subject, Math.max(0, reading.from - PER_PAGE));
});
el('later').addEventListener('click', () => {
  if (reading) openClaims(reading.subject, reading.from + PER_PAGE);
});

refresh();
