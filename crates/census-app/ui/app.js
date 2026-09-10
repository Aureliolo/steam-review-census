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
  if (game.stage !== 'classified') {
    panel.hidden = true;
    set(
      el('game-note'),
      'Downloaded but not counted yet. Reading a corpus turns it into rates you can open.',
    );
    return;
  }
  try {
    const counted = await invoke('analysis', { appId: game.app_id });
    if (counted.app_id !== chosen) return;
    drawTopics(counted);
    panel.hidden = false;
  } catch (failure) {
    panel.hidden = true;
    set(el('game-note'), String(failure));
  }
}

function drawTopics(counted) {
  const ranked = counted.topics
    .filter((topic) => topic.mentions > 0)
    .sort((left, right) => right.mentions - left.mentions);
  const widest = ranked.length > 0 ? (ranked[0].rate ?? 0) : 0;

  set(
    el('topics-lede'),
    counted.positive_baseline === null
      ? `${whole.format(counted.reviews)} reviews counted.`
      : `${whole.format(counted.reviews)} reviews counted, ${share.format(
          counted.positive_baseline,
        )} of them recommending the game. Every rate below opens into the reviews behind it.`,
  );

  const rows = el('topic-rows');
  rows.replaceChildren();
  for (const topic of ranked) {
    const row = document.createElement('tr');

    const subject = document.createElement('td');
    const open = document.createElement('button');
    open.type = 'button';
    open.className = 'subject';
    open.textContent = topic.label;
    open.addEventListener('click', () => openBehind(topic, 0));
    subject.append(open);

    const rate = document.createElement('td');
    rate.className = 'num rate';
    const reading = document.createElement('span');
    reading.textContent = topic.rate === null ? '—' : share.format(topic.rate);
    const bar = document.createElement('i');
    bar.className = 'bar';
    bar.style.transform = `scaleX(${widest > 0 ? (topic.rate ?? 0) / widest : 0})`;
    rate.append(reading, bar);

    const recommend = document.createElement('td');
    recommend.className = 'num';
    const liked = document.createElement('span');
    liked.textContent = topic.positive === null ? '—' : share.format(topic.positive);
    if (counted.positive_baseline !== null && topic.positive !== null) {
      liked.className = topic.positive < counted.positive_baseline ? 'over' : 'under';
    }
    recommend.append(liked);

    const top = document.createElement('td');
    top.className = 'num';
    const gauge = document.createElement('span');
    if (topic.bias === null) {
      gauge.textContent = '—';
      gauge.className = 'faint';
    } else {
      gauge.textContent = `${topic.bias.toFixed(1)}×`;
      if (topic.bias >= 1.15) gauge.className = 'over';
      else if (topic.bias <= 0.87) gauge.className = 'under';
      else gauge.className = 'faint';
    }
    top.append(gauge);

    row.append(subject, rate, recommend, top);
    rows.append(row);
  }
}

let reading = null;

async function openBehind(topic, from) {
  reading = { topic, from };
  show('evidence');
  set(el('evidence-name'), topic.label);
  set(el('evidence-lede'), 'Finding them...');
  el('quotes').replaceChildren();
  el('earlier').disabled = true;
  el('later').disabled = true;

  let page;
  try {
    page = await invoke('behind', {
      appId: chosen,
      category: topic.id,
      from,
      count: PER_PAGE,
    });
  } catch (failure) {
    set(el('evidence-lede'), String(failure));
    return;
  }
  if (reading?.topic.id !== topic.id || reading.from !== from) return;

  set(
    el('evidence-lede'),
    `${whole.format(page.total)} reviews raise this, ${
      topic.rate === null ? '' : `${share.format(topic.rate)} of the corpus, `
    }shown in the order they were written.`,
  );
  drawQuotes(page.quotes);

  const upTo = from + page.quotes.length;
  set(el('paging-note'), `${whole.format(from + 1)} to ${whole.format(upTo)}`);
  el('earlier').disabled = from === 0;
  el('later').disabled = upTo >= page.total;
}

function drawQuotes(quotes) {
  const list = el('quotes');
  list.replaceChildren();
  for (const quote of quotes) {
    const item = document.createElement('li');

    const body = document.createElement('p');
    body.textContent = quote.text;
    body.lang = bcp47(quote.language);

    const byline = document.createElement('div');
    byline.className = 'byline';

    const verdict = document.createElement('span');
    verdict.className = quote.voted_up ? 'verdict-up' : 'verdict-down';
    verdict.textContent = quote.voted_up ? 'Recommended' : 'Not recommended';
    byline.append(verdict);

    if (quote.votes_up > 0) {
      const votes = document.createElement('span');
      votes.textContent = `${whole.format(quote.votes_up)} found this helpful`;
      byline.append(votes);
    }

    const when = document.createElement('span');
    when.textContent = day.format(new Date(quote.created * 1000));
    byline.append(when);

    for (const mention of quote.mentions) {
      const tag = document.createElement('span');
      tag.className = 'tag';
      tag.textContent = mention;
      byline.append(tag);
    }

    if (quote.url) {
      const link = document.createElement('a');
      link.href = '#';
      link.textContent = 'On Steam';
      link.addEventListener('click', (event) => {
        event.preventDefault();
        openOutside(quote.url);
      });
      byline.append(link);
    }

    item.append(body, byline);
    list.append(item);
  }
}

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
el('earlier').addEventListener('click', () => {
  if (reading) openBehind(reading.topic, Math.max(0, reading.from - PER_PAGE));
});
el('later').addEventListener('click', () => {
  if (reading) openBehind(reading.topic, reading.from + PER_PAGE);
});

refresh();
