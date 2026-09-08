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
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setUp);
  } else {
    setUp();
  }
})();
