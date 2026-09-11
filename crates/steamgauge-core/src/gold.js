// The adjudication page. Everything is in the file already; nothing is fetched and nothing is
// sent. Answers are kept in this browser as they are made, because fourteen hundred claims is
// not one sitting and a closed tab must not cost a night's work.
(function () {
  "use strict";

  var data = JSON.parse(document.getElementById("data").textContent);
  var questions = data.questions || [];
  var categories = data.categories || [];
  var app = document.getElementById("app");
  var STORE = "steamgauge-gold-" + questions.length + "-" + (data.taxonomy || "");

  var answers = load();
  var at = firstUnanswered();

  function load() {
    try {
      return JSON.parse(localStorage.getItem(STORE) || "{}");
    } catch (whatever) {
      return {};
    }
  }

  function save() {
    try {
      localStorage.setItem(STORE, JSON.stringify(answers));
    } catch (whatever) {
      /* A full or blocked store is not a reason to stop; the export still works. */
    }
  }

  function keyOf(question) {
    return question.app_id + "#" + question.review_id + "#" + question.index;
  }

  function firstUnanswered() {
    for (var i = 0; i < questions.length; i += 1) {
      if (!answers[keyOf(questions[i])]) return i;
    }
    return questions.length;
  }

  function answered() {
    var n = 0;
    for (var i = 0; i < questions.length; i += 1) if (answers[keyOf(questions[i])]) n += 1;
    return n;
  }

  // Letters rather than numbers: twenty-six categories do not fit on the number row, and a
  // letter that matches the category's own name is the one a reader remembers.
  var keys = {};
  var taken = {};
  categories.forEach(function (category) {
    var want = category.id[0];
    var i = 0;
    while (taken[want] && i < category.id.length) {
      i += 1;
      want = category.id[i] || String.fromCharCode(97 + Object.keys(taken).length);
    }
    taken[want] = true;
    keys[category.id] = want;
  });

  function escape(text) {
    return String(text).replace(/[&<>"]/g, function (ch) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch];
    });
  }

  function marked(question) {
    // The claim is highlighted where it sits, rather than repeated: a reader asked to find the
    // sentence twice reads it once and guesses the second time.
    var start = question.at || 0;
    var end = start + (question.claim || "").length;
    var review = question.review || "";
    return (
      escape(review.slice(0, start)) +
      "<mark>" +
      escape(review.slice(start, end)) +
      "</mark>" +
      escape(review.slice(end))
    );
  }

  // The page redraws itself on every answer, so anything the reader opened has to be put back
  // or the sheet slams shut the moment they use it.
  var sheetOpen = false;

  function render() {
    if (at >= questions.length) return renderDone();
    var question = questions[at];
    var mine = answers[keyOf(question)] || {};

    var html = [];
    html.push('<header class="bar">');
    html.push("<h1>Adjudicate</h1>");
    html.push(
      '<span class="count">' +
        (at + 1) +
        " of " +
        questions.length +
        " &middot; " +
        answered() +
        " answered</span>"
    );
    html.push('<span class="spacer"></span>');
    html.push('<button class="go quiet" id="export">Export answers</button>');
    html.push("</header>");
    html.push(
      '<div class="progress"><i style="width:' +
        ((answered() / questions.length) * 100).toFixed(1) +
        '%"></i></div>'
    );

    html.push('<div class="card">');
    html.push('<p class="claim">' + escape(question.claim) + "</p>");
    html.push('<div class="review">' + marked(question) + "</div>");

    if (question.shown && question.shown.length) {
      html.push('<div class="shown"><span class="who">Two labellers split:</span>');
      question.shown.forEach(function (said, i) {
        html.push(
          "<span><b>" +
            escape(said.subject) +
            "</b> " +
            escape(said.polarity) +
            ' <span class="who">(' +
            escape(said.confidence) +
            (said.ambiguous ? ", called it contested" : "") +
            ")</span></span>"
        );
        if (i === 0) html.push('<span class="who">vs</span>');
      });
      html.push("</div>");
    }

    html.push('<div class="grid">');
    categories.forEach(function (category) {
      html.push(
        '<button class="pick' +
          (mine.subject === category.id ? " chosen" : "") +
          '" data-subject="' +
          escape(category.id) +
          '" title="' +
          escape(category.description || "") +
          '"><kbd>' +
          escape(keys[category.id]) +
          '</kbd><span class="what">' +
          escape(category.label) +
          "</span></button>"
      );
    });
    html.push("</div>");

    html.push('<div class="row"><span class="label">Polarity</span>');
    ["praise", "complaint", "neutral"].forEach(function (tone, i) {
      html.push(
        '<button class="tone' +
          (mine.polarity === tone ? " chosen" : "") +
          '" data-tone="' +
          tone +
          '">' +
          tone +
          " <kbd>" +
          (i + 1) +
          "</kbd></button>"
      );
    });
    html.push("</div>");

    html.push('<div class="row"><span class="label">This claim</span>');
    html.push(
      '<button class="tone' +
        (mine.ambiguous ? " chosen" : "") +
        '" id="ambiguous">is genuinely contested <kbd>0</kbd></button>'
    );
    html.push(
      '<button class="tone' +
        (mine.split_wrong ? " chosen" : "") +
        '" id="split">was cut wrong <kbd>9</kbd></button>'
    );
    // The reader's own uncertainty, kept apart from whether the claim is contested: one is
    // about them and one is about the sheet, and a set that conflates them cannot say which
    // of the two a disagreement came from.
    html.push(
      '<button class="tone' +
        (mine.unsure ? " chosen" : "") +
        '" id="unsure">I am unsure <kbd>8</kbd></button>'
    );
    html.push("</div>");
    html.push("</div>");

    html.push(
      '<p class="note"><kbd>&larr;</kbd> and <kbd>&rarr;</kbd> move, a letter picks a subject, ' +
        "<kbd>1</kbd>&ndash;<kbd>3</kbd> the polarity. Pick both and it moves on by itself. " +
        "<kbd>0</kbd> contested, <kbd>9</kbd> cut wrong, <kbd>8</kbd> unsure. " +
        "Your answers are kept in this browser as you go.</p>"
    );

    html.push(
      '<details class="sheet"' +
        (sheetOpen ? " open" : "") +
        "><summary>The category sheet</summary><dl>"
    );
    categories.forEach(function (category) {
      html.push("<dt>" + escape(category.label) + " <kbd>" + escape(keys[category.id]) + "</kbd></dt>");
      html.push("<dd>" + escape(category.description || "") + "</dd>");
      if (category.boundary) html.push("<dd><em>" + escape(category.boundary) + "</em></dd>");
    });
    html.push("</dl></details>");

    app.innerHTML = html.join("");
    wire(question);
  }

  function renderDone() {
    app.innerHTML =
      '<div class="done"><h2>' +
      answered() +
      " of " +
      questions.length +
      " answered</h2>" +
      '<p class="note">Export the file and run <code>steamgauge ingest-gold</code> on it.</p>' +
      '<p><button class="go" id="export">Export answers</button> ' +
      '<button class="go quiet" id="back">Back to the last one</button></p></div>';
    document.getElementById("export").onclick = exportAnswers;
    document.getElementById("back").onclick = function () {
      at = Math.max(0, questions.length - 1);
      render();
    };
  }

  function set(question, field, value) {
    var key = keyOf(question);
    var mine = answers[key] || {
      app_id: question.app_id,
      review_id: question.review_id,
      index: question.index,
    };
    mine[field] = value;
    answers[key] = mine;
    save();
    // Moving on the moment both halves of an answer exist is what makes fourteen hundred
    // claims possible: the reader never touches a "next" button.
    if (mine.subject && mine.polarity) {
      at += 1;
      render();
    } else {
      render();
    }
  }

  function wire(question) {
    Array.prototype.forEach.call(app.querySelectorAll("button.pick"), function (button) {
      button.onclick = function () {
        set(question, "subject", button.getAttribute("data-subject"));
      };
    });
    Array.prototype.forEach.call(app.querySelectorAll("button.tone[data-tone]"), function (button) {
      button.onclick = function () {
        set(question, "polarity", button.getAttribute("data-tone"));
      };
    });
    var mine = answers[keyOf(question)] || {};
    var ambiguous = document.getElementById("ambiguous");
    if (ambiguous) ambiguous.onclick = function () { set(question, "ambiguous", !mine.ambiguous); };
    var split = document.getElementById("split");
    if (split) split.onclick = function () { set(question, "split_wrong", !mine.split_wrong); };
    var unsure = document.getElementById("unsure");
    if (unsure) unsure.onclick = function () { set(question, "unsure", !mine.unsure); };
    var out = document.getElementById("export");
    if (out) out.onclick = exportAnswers;
    var sheet = app.querySelector("details.sheet");
    if (sheet) {
      sheet.ontoggle = function () {
        sheetOpen = sheet.open;
      };
    }
  }

  function exportAnswers() {
    var rows = [];
    questions.forEach(function (question) {
      var mine = answers[keyOf(question)];
      if (mine && mine.subject) rows.push(mine);
    });
    var blob = new Blob([JSON.stringify(rows, null, 2)], { type: "application/json" });
    var link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "gold-answers.json";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(link.href);
  }

  document.addEventListener("keydown", function (event) {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    if (at >= questions.length) return;
    var question = questions[at];
    if (event.key === "ArrowRight") {
      at = Math.min(questions.length, at + 1);
      render();
    } else if (event.key === "ArrowLeft") {
      at = Math.max(0, at - 1);
      render();
    } else if (event.key === "1" || event.key === "2" || event.key === "3") {
      set(question, "polarity", ["praise", "complaint", "neutral"][Number(event.key) - 1]);
    } else if (event.key === "0") {
      var mine = answers[keyOf(question)] || {};
      set(question, "ambiguous", !mine.ambiguous);
    } else if (event.key === "9") {
      var held = answers[keyOf(question)] || {};
      set(question, "split_wrong", !held.split_wrong);
    } else if (event.key === "8") {
      var doubted = answers[keyOf(question)] || {};
      set(question, "unsure", !doubted.unsure);
    } else {
      for (var id in keys) {
        if (keys[id] === event.key) {
          set(question, "subject", id);
          break;
        }
      }
    }
  });

  render();
})();
