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
  thing that suppresses SmartScreen and Gatekeeper warnings. **No build carries it.**
- **Self-signing.** Proves the bytes came from the holder of a key the operating system has
  never heard of. It changes no warnings.

So every build will warn on first launch. Windows will show a SmartScreen prompt you can
click past; macOS will refuse outright until you allow the app under System Settings,
Privacy and Security. That is expected, and it is not a signal that the download has been
tampered with. Verify provenance with the command above rather than relying on the
operating system's opinion, which here reflects only that nobody has paid a certificate
authority.

## Handling of credentials and data

The tool stores review corpora locally and never transmits them anywhere except, when a
hosted model is explicitly configured, the sampled subset sent to that provider for
labelling and summarisation. API keys are read from the environment or the local
configuration directory and are never written into a corpus, a log or a crash report.
