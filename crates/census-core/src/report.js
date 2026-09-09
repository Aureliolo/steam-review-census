// Inlined into the page. The report is readable with scripting off: every panel is a real
// table row and every long review is fully present in the markup, so this only adds folding.
(function () {
  'use strict';

  var root = document.documentElement;
  var STORED = 'census-theme';

  function applyStored() {
    try {
      var choice = localStorage.getItem(STORED);
      if (choice === 'dark' || choice === 'light') {
        root.setAttribute('data-theme', choice);
      }
    } catch (error) {
      // A browser refusing storage is not a reason to render nothing.
    }
  }

  function dark() {
    var chosen = root.getAttribute('data-theme');
    if (chosen) {
      return chosen === 'dark';
    }
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  }

  function toggleTheme(button) {
    var next = dark() ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    button.setAttribute('aria-pressed', String(next === 'dark'));
    try {
      localStorage.setItem(STORED, next);
    } catch (error) {
      // The page still switches; it just will not remember.
    }
  }

  // Each category row is followed by its own panel row, so the two move together or the
  // evidence ends up under somebody else's number.
  function pairs(body) {
    var out = [];
    var rows = body.querySelectorAll('tr.row');
    Array.prototype.forEach.call(rows, function (row) {
      var next = row.nextElementSibling;
      out.push([row, next && next.classList.contains('panel') ? next : null]);
    });
    return out;
  }

  function value(row, index) {
    var cell = row.children[index];
    if (!cell) {
      return 0;
    }
    var raw = cell.getAttribute('data-value');
    if (raw === null) {
      return cell.textContent.trim().toLowerCase();
    }
    return parseFloat(raw);
  }

  function sortable(table) {
    var body = table.tBodies[0];
    var buttons = table.querySelectorAll('thead [data-sort]');
    Array.prototype.forEach.call(buttons, function (button, index) {
      button.addEventListener('click', function () {
        var header = button.parentNode;
        // A first click on a number sorts largest first, which is what anyone reading a
        // rate wants; a first click on a name sorts A to Z.
        var numeric = header.classList.contains('num');
        var current = header.getAttribute('aria-sort');
        var descending = current === 'none' ? numeric : current === 'ascending';

        var sorted = pairs(body).sort(function (left, right) {
          var a = value(left[0], index);
          var b = value(right[0], index);
          if (a === b) {
            return 0;
          }
          var order = a > b ? 1 : -1;
          return descending ? -order : order;
        });

        Array.prototype.forEach.call(table.querySelectorAll('thead th'), function (th) {
          th.setAttribute('aria-sort', 'none');
        });
        header.setAttribute('aria-sort', descending ? 'descending' : 'ascending');

        sorted.forEach(function (pair) {
          body.appendChild(pair[0]);
          if (pair[1]) {
            body.appendChild(pair[1]);
          }
        });
      });
    });
  }

  // Every row that names a category, in the cross-game matrix and in each game's table, with
  // the evidence panel that has to move with it.
  function named() {
    var out = [];
    var tables = document.querySelectorAll('table.categories, table.matrix');
    Array.prototype.forEach.call(tables, function (table) {
      var body = table.tBodies[0];
      if (!body) {
        return;
      }
      Array.prototype.forEach.call(body.rows, function (row) {
        // The rendered cell carries a warning mark and text meant only for a screen
        // reader, so the name is read off the row rather than out of it.
        var name = row.getAttribute('data-name');
        if (name === null) {
          return;
        }
        var next = row.nextElementSibling;
        out.push({
          row: row,
          panel: next && next.classList.contains('panel') ? next : null,
          name: name
        });
      });
    });
    return out;
  }

  function narrow(form) {
    var input = form.querySelector('[data-filter-input]');
    var count = form.querySelector('[data-filter-count]');
    var rows = named();
    var total = {};
    rows.forEach(function (entry) {
      total[entry.name] = true;
    });
    var all = Object.keys(total).length;

    function apply() {
      var query = input.value.trim().toLowerCase();
      var matched = {};
      rows.forEach(function (entry) {
        var hit = query === '' || entry.name.indexOf(query) !== -1;
        // A class rather than the hidden attribute, which already means "folded away" on
        // these rows. Printing opens everything folded and must not undo a filter with it.
        entry.row.classList.toggle('filtered-out', !hit);
        if (entry.panel) {
          entry.panel.classList.toggle('filtered-out', !hit);
        }
        if (hit) {
          matched[entry.name] = true;
        }
      });
      var shown = Object.keys(matched).length;
      if (query === '') {
        count.textContent = 'all ' + all + ' of them';
      } else if (shown === 0) {
        count.textContent = 'no category matches';
      } else {
        count.textContent = shown + ' of ' + all;
      }
    }

    input.addEventListener('input', apply);
    // A search input clears itself, and browsers disagree about which event says so.
    input.addEventListener('search', apply);
    form.addEventListener('submit', function (event) {
      event.preventDefault();
    });
    form.hidden = false;
  }

  // Nobody can open a fold on paper. The stylesheet handles the rows and the clipped text,
  // but a closed <details> is hidden by the browser itself and no author rule reaches it.
  function unfoldToPrint() {
    var reopened = [];
    window.addEventListener('beforeprint', function () {
      reopened = [];
      Array.prototype.forEach.call(document.querySelectorAll('details'), function (fold) {
        if (!fold.open) {
          fold.open = true;
          reopened.push(fold);
        }
      });
    });
    window.addEventListener('afterprint', function () {
      reopened.forEach(function (fold) {
        fold.open = false;
      });
      reopened = [];
    });
  }

  // A link to one category's evidence has to arrive with that evidence open, or it lands on
  // a row that looks exactly like every other row.
  function openWhatTheAddressAsksFor() {
    var id = window.location.hash.replace('#', '');
    if (!id) {
      return;
    }
    var panel = document.getElementById(id);
    if (!panel || !panel.classList.contains('panel')) {
      return;
    }
    var row = panel.previousElementSibling;
    if (row && row.getAttribute('data-expands') === id) {
      row.setAttribute('aria-expanded', 'true');
    }
    panel.hidden = false;
    panel.scrollIntoView();
  }

  function setUp() {
    // Tells the stylesheet that folding is available. Everything folded is fully present in
    // the markup, so a page without scripting is longer rather than incomplete.
    root.classList.add('js');
    applyStored();

    var toggle = document.querySelector('[data-theme-toggle]');
    if (toggle) {
      toggle.setAttribute('aria-pressed', String(dark()));
      toggle.addEventListener('click', function () {
        toggleTheme(toggle);
      });
    }

    // Panels start closed only once scripting is known to work. Without this a reader with
    // no scripting would face rows that can never be opened.
    var rows = document.querySelectorAll('[data-expands]');
    Array.prototype.forEach.call(rows, function (row) {
      var panel = document.getElementById(row.getAttribute('data-expands'));
      if (!panel) {
        return;
      }
      panel.hidden = true;
      row.setAttribute('aria-expanded', 'false');

      function toggle() {
        var open = row.getAttribute('aria-expanded') === 'true';
        row.setAttribute('aria-expanded', String(!open));
        panel.hidden = open;
      }

      row.addEventListener('click', toggle);
      row.addEventListener('keydown', function (event) {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          toggle();
        }
      });
    });

    Array.prototype.forEach.call(
      document.querySelectorAll('table.categories'),
      sortable
    );

    var mores = document.querySelectorAll('[data-expands-text]');
    Array.prototype.forEach.call(mores, function (button) {
      var text = button.previousElementSibling;
      if (!text) {
        return;
      }
      button.addEventListener('click', function () {
        var open = text.classList.toggle('open');
        button.setAttribute('aria-expanded', String(open));
        button.textContent = open ? 'Show less' : 'Show the rest';
      });
    });

    // After the panels, which decide for themselves whether they start open.
    var form = document.querySelector('[data-filter]');
    if (form && form.querySelector('[data-filter-input]')) {
      narrow(form);
    }

    unfoldToPrint();
    openWhatTheAddressAsksFor();
    window.addEventListener('hashchange', openWhatTheAddressAsksFor);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setUp);
  } else {
    setUp();
  }
})();
