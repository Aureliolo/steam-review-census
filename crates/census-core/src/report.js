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
