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
