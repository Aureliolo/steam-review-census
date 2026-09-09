# Contributing

The project is not usable yet and the approach is still moving. Before starting anything
substantial, open an issue so the design is settled first.

## Working on it

`main` is protected. Everything lands through a pull request, including maintainer changes
and dependency bumps. Branches must be current with `main` before merging, and history is
linear, so merges are squashed or rebased rather than committed as merges.

Run the same gates CI runs before pushing:

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

All three run on Linux, macOS and Windows. Clippy warnings fail the build.

The report page carries scripting no Rust test can reach, and rendering no Rust test can see:
a stylesheet rule can flatten a chart or turn a printed page into blocks of ink while the
markup stays exactly right. A fourth gate therefore drives the page in a real browser three
times: on a desktop window, at 420 pixels where the tables are wider than the screen, and
under print media with the machine asking for a dark one. It needs Chrome and Node, and
nothing else: no corpus, no model, no network.

```sh
cargo run -p census-core --example sample-report -- report.html
node tools/report-check/check.mjs report.html
```

Chrome is found in the usual places per platform, or wherever `CHROME_PATH` says. Every
check in it is a promise the page makes in its own prose; if you change what the page says
it does, change the check with it.

## What is held to a higher bar

This tool exists to make a percentage mean what it appears to mean, so anything affecting a
reported number needs more than passing tests:

- **Ingestion parameters.** The census depends on overriding Valve's defaults. A change here
  silently changes every downstream figure and can make existing corpora incomparable.
- **Denominators.** A percentage is a mention rate unless labelled otherwise. Do not add a
  figure to the interface or the README without stating which denominator it uses.
- **Taxonomy, classifier and gold set.** Changing any of these changes the numbers. Say by
  how much.
- **Claims in the README.** Every figure quoted must be reproducible, and the pull request
  should say how to reproduce it.

## Licence

Contributions are accepted under the Apache License 2.0, as covered by section 5 of the
licence.
