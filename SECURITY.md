# Security

## Reporting a vulnerability

Report privately through GitHub's [security advisory
form](https://github.com/Aureliolo/steam-review-census/security/advisories/new). Please do
not open a public issue for a vulnerability.

There is no release yet, so there is nothing deployed to attack and no supported version to
patch. Reports about the build and release pipeline are in scope and welcome.

## Verifying a release

Every release is built by GitHub Actions from a tagged commit, published as an immutable
release, and carries a SLSA build-provenance attestation. Nothing is built or signed on a
developer machine.

Verify an asset before running it:

```sh
gh attestation verify <asset> --repo Aureliolo/steam-review-census
```

Checksums for every asset are published as `SHA256SUMS` alongside the release.

## What signing does and does not tell you

Three different things are commonly called signing, and they answer different questions:

- **Provenance** (SLSA attestations, cosign). Proves an asset was built by this repository's
  workflow from a specific commit. Verified deliberately, with the command above.
- **Operating-system trust** (a certificate chaining to a CA the OS ships). This is the only
  thing that suppresses SmartScreen and Gatekeeper warnings. Windows builds carry it;
  macOS builds do not.
- **Self-signing.** Proves the bytes came from the holder of a key the operating system has
  never heard of. It changes no warnings.

macOS builds are unsigned by Apple's definition, so Gatekeeper will refuse them on first
launch and you will need to allow the app explicitly under System Settings, Privacy and
Security. That is expected, and it is not a signal that the download is untrustworthy.
Verify provenance with the command above rather than relying on the operating system's
opinion.

## Handling of credentials and data

The tool stores review corpora locally and never transmits them anywhere except, when a
hosted model is explicitly configured, the sampled subset sent to that provider for
labelling and summarisation. API keys are read from the environment or the local
configuration directory and are never written into a corpus, a log or a crash report.
