"""Every training run on one set of labels, ranked, with the interval around each figure.

A sweep produces one number per configuration and the temptation is to read the largest one as
the winner. Most of the gaps are smaller than the noise: 2,615 validation claims put about two
points of uncertainty on any coverage figure, so two runs a point apart are the same run. This
prints the interval beside the figure so a choice between configurations is made against it,
and it refuses to put two runs in one table unless they were trained on the same labels, since
a run on more labels is not a better configuration.

    python sweep.py                     # every run on the current labels, best first
    python sweep.py --against win-128   # what each run changed, and what it bought
    python sweep.py --subjects lr5e5    # where one run wins and loses, subject by subject

Coverage is the headline because the promise is fixed: the reader answers at the accuracy it
promised or it abstains, so the configuration that answers more claims at the same promise is
the better one.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Everything `train.py` records that describes the configuration rather than the outcome, and
# for each the behaviour a run that predates the setting had. The seed is not here: it is not
# a configuration but a draw from one, and it is reported as a spread instead.
SETTINGS = {
    "backbone": None,
    "epochs": None,
    "batch_size": None,
    "learning_rate": None,
    "max_length": None,
    "context": None,
    "mark": False,
    "balance": 0.0,
    "ambiguous_weight": 1.0,
    "split_wrong_weight": 1.0,
    "polarity_weight": 0.5,
}


def wilson(hits: float, total: int, z: float = 1.96) -> tuple[float, float]:
    """The interval around a proportion that stays inside 0 and 1 near the ends."""
    if total <= 0:
        return (0.0, 1.0)
    share = hits / total
    denominator = 1 + z * z / total
    centre = (share + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(share * (1 - share) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def half_width(share: float, total: int) -> float:
    low, high = wilson(share * total, total)
    return (high - low) / 2


def runs_on(directory: Path) -> list[dict]:
    found = []
    for record in sorted(directory.glob("*/run.json")):
        with record.open(encoding="utf-8") as handle:
            run = json.load(handle)
        run["id"] = record.parent.name
        found.append(run)
    return found


def setting(run: dict, key: str):
    """What the run was configured as, taking a run older than the setting at its default."""
    if key in run:
        return run[key]
    return SETTINGS[key]


def shown(key: str, value) -> str:
    if value is None:
        return f"{key.replace('_', ' ')} unrecorded"
    if key == "backbone":
        value = str(value).split("/")[-1]
    if key == "learning_rate":
        value = f"{value:.0e}".replace("e-0", "e-")
    return f"{key.replace('_', ' ')} {value}"


def differences(run: dict, against: dict) -> str:
    changed = [
        shown(key, setting(run, key))
        for key in SETTINGS
        if setting(run, key) != setting(against, key)
    ]
    return ", ".join(changed) or "the baseline"


def family(run: dict) -> tuple:
    """Everything but the seed, so that two runs of one configuration land together."""
    return tuple(str(setting(run, key)) for key in SETTINGS)


def table(found: list[dict], against: dict | None) -> None:
    rows = []
    for run in found:
        result = run["validation"]
        claims = run["claims"]["validation"]
        coverage = result["threshold_coverage"]
        answered = max(1, round(coverage * claims))
        rows.append(
            {
                "id": run["id"],
                "what": differences(run, against) if against else "",
                "coverage": coverage,
                "coverage_error": half_width(coverage, claims),
                "accuracy": result["threshold_accuracy"],
                "accuracy_error": half_width(result["threshold_accuracy"], answered),
                "macro_f1": result["macro_f1"],
                "aurc": result["aurc"],
                "ece": result["calibration_error"],
                "met": result["threshold_met"],
                "minutes": run["seconds"] / 60,
            }
        )
    rows.sort(key=lambda row: row["coverage"], reverse=True)

    width = max(len(row["id"]) for row in rows)
    what = max((len(row["what"]) for row in rows), default=0)
    header = f"{'run':<{width}}  "
    if against:
        header += f"{'what it changed':<{what}}  "
    print(header + "answers            at accuracy        macro F1   AURC    ECE    min")
    best = rows[0]
    for row in rows:
        mark = " " if row["met"] else "!"
        line = f"{row['id']:<{width}}{mark} "
        if against:
            line += f"{row['what']:<{what}}  "
        # A run whose interval overlaps the best run's is not distinguishable from it, and
        # saying so is the whole point of printing the interval.
        same = row["coverage"] + row["coverage_error"] >= best["coverage"] - best["coverage_error"]
        line += (
            f"{row['coverage']:6.1%} +-{row['coverage_error']:4.1%}{'=' if same else ' '} "
            f"{row['accuracy']:6.1%} +-{row['accuracy_error']:4.1%}  "
            f"{row['macro_f1']:8.3f} {row['aurc']:6.3f} {row['ece']:6.3f} {row['minutes']:5.0f}"
        )
        print(line)
    print()
    print("! the run never reached the accuracy it promised at any threshold")
    print("= its coverage is inside the best run's interval, so it is not a worse run")
    print("+- is the sampling interval on one model's answers, not the spread between runs")


def spread(found: list[dict], against: dict | None) -> None:
    """How far apart two runs of one configuration land, which is the bar a gap has to clear.

    The interval beside a coverage figure is the noise in scoring one model on 2,615 claims.
    It says nothing about training the same configuration twice, and that is much noisier: a
    different initialisation of the two heads and a different order through the data land
    several points apart. A configuration change worth taking has to beat this, not that.
    """
    families: dict[tuple, list[dict]] = {}
    for run in found:
        # A run from before the seed was recorded was not seeded at all, so it is a draw from
        # the configuration rather than a repeat of a named one. It cannot join the spread:
        # nothing says the code it ran was the code the others ran.
        if "seed" in run:
            families.setdefault(family(run), []).append(run)
    repeated = [runs for runs in families.values() if len(runs) > 1]
    if not repeated:
        print("\nno configuration was trained on more than one seed, so the spread is unmeasured")
        return

    print("\nthe same configuration on another seed:")
    widest = 0.0
    for runs in sorted(repeated, key=len, reverse=True):
        coverages = [run["validation"]["threshold_coverage"] for run in runs]
        gap = max(coverages) - min(coverages)
        widest = max(widest, gap)
        names = ", ".join(
            f"{run['id']} {run['validation']['threshold_coverage']:.1%}"
            for run in sorted(runs, key=lambda run: -run["validation"]["threshold_coverage"])
        )
        what = differences(runs[0], against) if against else ""
        print(f"  {gap:5.1%} apart: {names}" + (f"   ({what})" if what else ""))
    print(f"\na configuration has to beat {widest:.1%} of coverage to have changed anything")


def subjects(run: dict, against: dict | None) -> None:
    mine = run["validation"]["per_subject"]
    theirs = against["validation"]["per_subject"] if against else {}
    width = max(len(subject) for subject in mine)
    print(f"{'subject':<{width}}  labels       F1" + ("   against" if theirs else ""))
    for subject, scores in sorted(mine.items(), key=lambda pair: -pair[1]["f1"]):
        line = f"{subject:<{width}}  {scores['support']:6d} {scores['f1']:8.3f}"
        if subject in theirs:
            line += f"  {scores['f1'] - theirs[subject]['f1']:+8.3f}"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=HERE / "runs")
    parser.add_argument(
        "--against",
        help="the run whose configuration every other run is described as a change from",
    )
    parser.add_argument("--subjects", help="print one run's per-subject scores instead")
    parser.add_argument(
        "--fingerprint",
        help="which labels to report on. The set the most runs were trained on when omitted.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="report every run, including ones trained on other labels",
    )
    arguments = parser.parse_args()

    found = runs_on(arguments.runs)
    if not found:
        raise SystemExit(f"no runs under {arguments.runs}")

    fingerprint = arguments.fingerprint
    if not fingerprint and not arguments.all:
        fingerprint = Counter(run["data_fingerprint"] for run in found).most_common(1)[0][0]
    if fingerprint:
        skipped = [run for run in found if run["data_fingerprint"] != fingerprint]
        found = [run for run in found if run["data_fingerprint"] == fingerprint]
        if not found:
            raise SystemExit(f"no runs on labels {fingerprint}")
        print(
            f"{len(found)} runs on labels {fingerprint}: "
            f"{found[0]['claims']['train']} training claims, "
            f"{found[0]['claims']['validation']} validation claims"
        )
        if skipped:
            print(f"{len(skipped)} runs on other labels are not comparable and are left out")
        print()

    by_id = {run["id"]: run for run in found}
    against = None
    if arguments.against:
        against = by_id.get(arguments.against)
        if against is None:
            raise SystemExit(f"no run {arguments.against} on these labels")

    if arguments.subjects:
        run = by_id.get(arguments.subjects)
        if run is None:
            raise SystemExit(f"no run {arguments.subjects} on these labels")
        subjects(run, against)
        return

    table(found, against)
    spread(found, against)


if __name__ == "__main__":
    main()
