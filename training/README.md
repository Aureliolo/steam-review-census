# Training

The classifier is not written in Rust. It is trained here, in Python, and shipped as an ONNX
graph the Rust tool runs. Nothing in this directory is part of the build; everything in it
produces an artefact that is, and every artefact records what produced it.

## What it does

`census` labels claims, not reviews: the separate points a review makes. A model is trained to
read one claim and say which of the core-spine subjects it is about, whether it is praise,
complaint or neither, and how sure it is. Below a calibrated threshold it says nothing, which
is the whole reason for replacing what came before: the previous classifier compared a review
to twenty-four category prototypes and took the nearest, so a review reading "gfg" was filed
under Graphics and art.

## Setup

```sh
python -m venv .venv
.venv/Scripts/activate          # or .venv/bin/activate
pip install -r requirements.txt
pip freeze > requirements.lock  # what actually resolved, committed beside this
```

Training wants a GPU. It will run on a CPU and you will not enjoy it.

## The order things happen in

```sh
census export-training --to training/data/claims.jsonl   # labels plus their text, local only
python bakeoff.py                                        # which backbone, decided by measurement
python train.py --backbone <the winner> --save           # the model
python export.py --run runs/<id> --spine core-5 --fp16   # ONNX, with a parity assertion
python publish.py --run runs/<id> \
    --model-repo <you>/steam-review-claim-reader \
    --data-repo <you>/steam-review-claims                # to Hugging Face, and pinned in the tool
```

`claims.jsonl` holds review text and is never committed. What gets published is the model and a
label set of review ids, claim offsets and labels, built from `reference/claims/` rather than
from the training export, which anyone can rehydrate with `census` itself. Reviews belong to
the people who wrote them, and `publish.py` refuses a row that carries text.

Publishing pins the three model files by hash in `crates/census-core/src/reader.rs`, so the
tool fetches exactly what went up and refuses anything else. Rebuild and commit after it runs.
Until something is published the pin is empty, and an empty pin refuses to fetch rather than
fetching unverified.

## What is recorded

Every run writes `runs/<id>/` holding the config, the git sha, a hash of the label set, which
games were trained on, validated on and frozen, the metrics per subject and per language, the
whole risk-coverage curve, and the frozen games' figures at the threshold validation chose.
That directory is the only evidence a published figure has, so a run that did not write one
did not happen.

The frozen games are read once, at the end, and choose nothing. The bake-off does not read
them at all: a set used to choose between models cannot also say how the chosen one does.
