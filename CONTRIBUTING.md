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
markup stays exactly right. A fourth gate therefore drives the page in a real browser four
times: on a desktop window, at 420 pixels where the tables are wider than the screen, under
print media with the machine asking for a dark one, and again with the script cut out, which
is what a reader with scripting off is served. It needs Chrome and Node, and nothing else: no
corpus, no model, no network.

```sh
cargo run -p census-core --example sample-report -- report.html
node tools/report-check/check.mjs report.html
```

Chrome is found in the usual places per platform, or wherever `CHROME_PATH` says. Every
check in it is a promise the page makes in its own prose; if you change what the page says
it does, change the check with it.

### One build directory

`target/`, and nothing beside it. The gates above build test binaries under `target/debug` and
never produce a release one, so the only binary anyone runs for real work is
`target/release/census.exe`. Where there is a GPU, build it with the backend for it, because
embedding a million reviews on a CPU is the difference between an afternoon and a week:

```sh
cargo build --release -p census-cli --features directml
```

Two things follow. A build without that flag replaces the same file with a CPU one, and the
only sign is the device `census embed` prints on its third line: read it. And on Windows a
running `census.exe` holds its own binary open, so a build started mid-crawl fails with
"Access is denied" — wait for the run rather than reaching for `--target-dir`, which is how
this repository once ended up with four build trees and twenty gigabytes in them.

Two of those promises are watched rather than read. The page is reloaded with the network
being listened to and fails on any request but the file itself, because a stylesheet pulling
a font would pass any amount of reading the markup for URLs. And every piece of text on it is
measured against whatever is composited behind it, in both themes, because a palette is a set
of tokens until a browser draws it.

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
