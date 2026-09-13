// The adjudication page. Every question is in the file already and nothing is fetched to ask
// one. Answers are kept in this browser as they are made, because fourteen hundred claims is
// not one sitting and a closed tab must not cost a night's work.
//
// Opened from a file, the browser is the only copy until somebody presses Export, and a browser
// loses its storage for reasons nobody controls. Served by `steamgauge gold --serve`, every
// answer is posted as it is made and the disk is the copy that matters: the page reads the file
// back when it opens, the store here becomes a cache, and there is nothing to remember.
(function () {
  "use strict";

  var data = JSON.parse(document.getElementById("data").textContent);
  var questions = data.questions || [];
  var categories = data.categories || [];
  var app = document.getElementById("app");
  var STORE = "steamgauge-gold-" + questions.length + "-" + (data.taxonomy || "");
  var SERVED = location.protocol === "http:" || location.protocol === "https:";

  var answers = load();
  var at = firstUnanswered();
  var notice = "";
  var kept = SERVED ? "saved" : "";

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
      /* A full or blocked store is not a reason to stop; the file on disk is the copy. */
    }
    if (SERVED) post();
  }

  // Posted whole rather than as a delta. A thousand answers is a few hundred kilobytes to a
  // process on the same machine, and sending the lot means a dropped request costs nothing: the
  // next answer carries everything the missed one did.
  function post() {
    var rows = exportable();
    fetch("answers", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(rows),
    })
      .then(function (reply) {
        kept = reply.ok ? "saved" : "not saved";
        paintKept();
      })
      .catch(function () {
        kept = "not saved";
        paintKept();
      });
  }

  function paintKept() {
    var where = document.getElementById("kept");
    if (where) where.textContent = kept;
  }

  // What is on disk wins over what this browser remembers: the file is the one copy that
  // survives a cleared cache and a second machine, and a browser holding a stale half of the
  // work must not overwrite it.
  function adopt(rows) {
    var wanted = {};
    questions.forEach(function (question) {
      wanted[keyOf(question)] = true;
    });
    var taken = 0;
    rows.forEach(function (row) {
      var key = row.app_id + "#" + row.review_id + "#" + row.index;
      if (!wanted[key]) return;
      answers[key] = row;
      taken += 1;
    });
    if (taken) {
      at = firstUnanswered();
      render();
    }
  }

  // What the task is, in the page rather than in somebody's head. Without it the first screen
  // is a sentence, a review, twenty-six buttons and the word "polarity", and the commonest
  // thing to conclude from that is that the question is about the review. It is not: a review
  // is cut into its separate points and each one is its own question, so a review making
  // thirty-six points is thirty-six answers and not one.
  function howto() {
    return (
      '<details class="howto"' +
      (howtoOpen ? " open" : "") +
      "><summary>What you are doing</summary>" +
      "<p>Each question is about <b>one claim</b>: a single point somebody made. The review it " +
      "came from is shown underneath with the claim highlighted, because a claim like " +
      "&ldquo;it doesn&rsquo;t&rdquo; or &ldquo;same here&rdquo; means nothing on its own. " +
      "<b>Judge the highlighted claim, not the review around it.</b> A review that praises one " +
      "thing and complains about thirty-four others is thirty-five separate claims, and you " +
      "will be asked about them one at a time.</p>" +
      "<p>Two answers each time. <b>Which subject</b> the claim is about, from the sheet at the " +
      "foot of the page, and <b>what it says about that subject</b>: praise, a complaint, or " +
      "neither. Nothing else is being asked, so a claim naming no subject at all is " +
      "<b>Overall verdict</b> when it judges the game and <b>Not about the game</b> when it " +
      "does not.</p>" +
      "<p>Follow the sheet&rsquo;s rules where they disagree with your instinct: the rules are " +
      "what make one person&rsquo;s answers comparable with another&rsquo;s, and where a rule " +
      "is wrong that is worth finding out. If two subjects genuinely both fit and the sheet " +
      "does not settle it, press <kbd>0</kbd>, which is not the same as being unsure " +
      "(<kbd>8</kbd>).</p>" +
      "<p>You are not told which game it is, whether the reviewer recommended it, or what " +
      "anyone else answered. That is deliberate: an answer on the page is an answer in your " +
      "head, and a figure produced by agreeing with a suggestion measures nothing. Stopping " +
      "part-way is fine and the work is kept.</p>" +
      "</details>"
    );
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
  // letter out of the category's own name is the one a reader remembers. Each category takes
  // the first letter of its name nobody has taken, then the first letter of its label, then
  // any letter at all, because two categories sharing a key means one of them cannot be
  // reached from the keyboard and the reader finds that out at claim four hundred.
  var ALPHABET = "abcdefghijklmnopqrstuvwxyz";
  var keys = {};
  var taken = {};

  // Whichever category has fewest of its own letters left goes first, rather than whichever
  // comes first in the taxonomy. Assigning in taxonomy order lets `accessibility` and
  // `atmosphere` spend the letters `story` and `policy` needed, and those two then take
  // whatever the alphabet has left, which is nothing anyone remembers.
  var own = function (category) {
    return (category.id + " " + (category.label || "")).toLowerCase();
  };
  var free = function (category) {
    var count = 0;
    var seen = {};
    var letters = own(category);
    for (var i = 0; i < letters.length; i += 1) {
      var ch = letters[i];
      if (ALPHABET.indexOf(ch) >= 0 && !taken[ch] && !seen[ch]) {
        seen[ch] = true;
        count += 1;
      }
    }
    return count;
  };

  var waiting = categories.slice();
  while (waiting.length) {
    waiting.sort(function (a, b) {
      return free(a) - free(b);
    });
    var category = waiting.shift();
    var tried = own(category) + " " + ALPHABET;
    for (var i = 0; i < tried.length; i += 1) {
      var ch = tried[i];
      if (ALPHABET.indexOf(ch) >= 0 && !taken[ch]) {
        taken[ch] = true;
        keys[category.id] = ch;
        break;
      }
    }
  }

  function escape(text) {
    return String(text).replace(/[&<>"]/g, function (ch) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch];
    });
  }

  function marked(question) {
    // The claim is highlighted where it sits, rather than repeated: a reader asked to find the
    // sentence twice reads it once and guesses the second time. The review arrives already
    // split around the claim, so there is no offset here to be in the wrong units.
    return (
      escape(question.before || "") +
      "<mark>" +
      escape(question.claim || "") +
      "</mark>" +
      escape(question.after || "")
    );
  }

  // The page redraws itself on every answer, so anything the reader opened has to be put back
  // or the sheet slams shut the moment they use it.
  var sheetOpen = false;
  // Open until it is closed once, and closed thereafter: the explanation is needed on the
  // first question and in the way on the four hundredth.
  //
  // Deliberately not under `STORE`: that key holds the answers and only the answers, which is
  // an invariant the browser check asserts, and it is per-draw where this preference is not.
  var HOWTO = "steamgauge-adjudication-howto";
  var howtoOpen = (function () {
    try {
      return localStorage.getItem(HOWTO) !== "0";
    } catch (whatever) {
      return true;
    }
  })();

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
    if (SERVED) html.push('<span class="count" id="kept">' + escape(kept) + "</span>");
    html.push('<button class="go quiet" id="export">Export answers</button>');
    html.push(restoreControl());
    html.push("</header>");
    if (notice) html.push('<p class="note loaded">' + escape(notice) + "</p>");
    html.push(
      '<div class="progress"><i style="width:' +
        ((answered() / questions.length) * 100).toFixed(1) +
        '%"></i></div>'
    );

    html.push(howto());

    html.push('<div class="card">');
    html.push('<p class="asking">The claim you are judging</p>');
    html.push('<p class="claim">' + escape(question.claim) + "</p>");
    html.push('<p class="asking">The review it came from, so you can tell what it refers to</p>');
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
          escape(keys[category.id] || "") +
          '</kbd><span class="what">' +
          escape(category.label) +
          "</span></button>"
      );
    });
    html.push("</div>");

    // Before the polarity, not after it. Answering both halves moves the page on, so a row
    // below the polarity is a row the reader never reaches without going back for it.
    html.push('<div class="row"><span class="label">Anything else? (optional)</span>');
    html.push(
      '<button class="tone' +
        (mine.ambiguous ? " chosen" : "") +
        '" id="ambiguous">two subjects both fit <kbd>0</kbd></button>'
    );
    html.push(
      '<button class="tone' +
        (mine.split_wrong ? " chosen" : "") +
        '" id="split">this is two points, or half of one <kbd>9</kbd></button>'
    );
    // The reader's own uncertainty, kept apart from whether the claim is contested: one is
    // about them and one is about the sheet, and a set that conflates them cannot say which
    // of the two a disagreement came from.
    html.push(
      '<button class="tone' +
        (mine.unsure ? " chosen" : "") +
        '" id="unsure">I am not sure <kbd>8</kbd></button>'
    );
    html.push("</div>");

    var chosen = mine.subject
      ? (categories.filter(function (one) {
          return one.id === mine.subject;
        })[0] || {}).label
      : null;
    html.push(
      '<div class="row last"><span class="label">Does the claim praise ' +
        (chosen ? "<b>" + escape(chosen.toLowerCase()) + "</b>" : "that") +
        " or complain about it?</span>"
    );
    [
      ["praise", "praises it"],
      ["complaint", "complains about it"],
      ["neutral", "neither"],
    ].forEach(function (pair, i) {
      html.push(
        '<button class="tone' +
          (mine.polarity === pair[0] ? " chosen" : "") +
          '" data-tone="' +
          pair[0] +
          '">' +
          pair[1] +
          " <kbd>" +
          (i + 1) +
          "</kbd></button>"
      );
    });
    html.push("</div>");
    html.push("</div>");

    var flagged = mine.ambiguous || mine.split_wrong || mine.unsure;
    html.push(
      '<p class="note"><kbd>&larr;</kbd> and <kbd>&rarr;</kbd> move, a letter picks a subject, ' +
        "<kbd>1</kbd>&ndash;<kbd>3</kbd> praise, complaint or neither. " +
        (flagged
          ? "This one is flagged, so it waits for <kbd>&rarr;</kbd> rather than moving on by " +
            "itself."
          : "Pick both and it moves on by itself.") +
        " <kbd>0</kbd> two subjects fit, <kbd>9</kbd> cut wrong, <kbd>8</kbd> not sure. " +
        (SERVED
          ? "Every answer is written to disk as you make it."
          : "Your answers are kept in this browser as you go.") +
        "</p>"
    );

    html.push(
      '<details class="sheet"' +
        (sheetOpen ? " open" : "") +
        "><summary>The category sheet</summary><dl>"
    );
    categories.forEach(function (category) {
      html.push("<dt>" + escape(category.label) + " <kbd>" + escape(keys[category.id] || "") + "</kbd></dt>");
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
      '<p class="note">' +
      (SERVED
        ? "Already on disk. Run <code>steamgauge ingest-gold</code> on it."
        : "Export the file and run <code>steamgauge ingest-gold</code> on it.") +
      "</p>" +
      '<p><button class="go" id="export">Export answers</button> ' +
      restoreControl() +
      ' <button class="go quiet" id="back">Back to the last one</button></p>' +
      (notice ? '<p class="note loaded">' + escape(notice) + "</p>" : "") +
      "</div>";
    document.getElementById("export").onclick = exportAnswers;
    wireRestore();
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
    //
    // Except on a claim the reader has flagged. The flags are optional and the rows are
    // ordered so they come before the polarity, but somebody who decides a claim is contested
    // after answering it would otherwise watch the page leave while they reached for the key,
    // and those are the claims the whole exercise is for. A flagged claim waits for an arrow.
    var flagged = mine.ambiguous || mine.split_wrong || mine.unsure;
    if (mine.subject && mine.polarity && !flagged) {
      at += 1;
    }
    render();
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
    wireRestore();
    var sheet = app.querySelector("details.sheet");
    if (sheet) {
      sheet.ontoggle = function () {
        sheetOpen = sheet.open;
      };
    }
    var how = app.querySelector("details.howto");
    if (how) {
      how.ontoggle = function () {
        howtoOpen = how.open;
        try {
          localStorage.setItem(HOWTO, howtoOpen ? "1" : "0");
        } catch (whatever) {
          /* Closing the instructions is a convenience, not work to be kept. */
        }
      };
    }
  }

  // The browser is the only copy of the work until the file is exported, and a browser loses
  // its storage for reasons nobody controls: a cleared cache, a private window, another
  // machine. Reading an exported file back turns the export into a save rather than a delivery.
  function restoreControl() {
    return (
      '<label class="go quiet" for="restore">Load answers</label>' +
      '<input type="file" id="restore" accept="application/json,.json" hidden>'
    );
  }

  function wireRestore() {
    var input = document.getElementById("restore");
    if (!input) return;
    input.onchange = function () {
      var file = input.files && input.files[0];
      if (file) restore(file);
    };
  }

  function restore(file) {
    var reader = new FileReader();
    reader.onload = function () {
      var rows;
      try {
        rows = JSON.parse(String(reader.result));
      } catch (whatever) {
        rows = null;
      }
      if (!rows || !rows.length) {
        notice = "that file holds no answers";
        return render();
      }
      var wanted = {};
      questions.forEach(function (question) {
        wanted[keyOf(question)] = true;
      });
      var taken = 0;
      var strangers = 0;
      rows.forEach(function (row) {
        if (!row || !row.subject) return;
        var key = row.app_id + "#" + row.review_id + "#" + row.index;
        // A file from another draw would fill the store with answers to questions this page
        // never asks, and the count in the corner would climb while nothing got adjudicated.
        if (!wanted[key]) {
          strangers += 1;
          return;
        }
        answers[key] = row;
        taken += 1;
      });
      save();
      at = firstUnanswered();
      notice =
        taken + " answers loaded" + (strangers ? ", " + strangers + " from another draw ignored" : "");
      render();
    };
    reader.readAsText(file);
  }

  function exportable() {
    var rows = [];
    questions.forEach(function (question) {
      var mine = answers[keyOf(question)];
      if (mine && mine.subject) rows.push(mine);
    });
    return rows;
  }

  function exportAnswers() {
    var blob = new Blob([JSON.stringify(exportable(), null, 2)], { type: "application/json" });
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

  if (SERVED) {
    fetch("answers")
      .then(function (reply) {
        return reply.ok ? reply.json() : [];
      })
      .then(function (rows) {
        if (rows && rows.length) adopt(rows);
      })
      .catch(function () {
        kept = "not saved";
        paintKept();
      });
  }
})();
