# Claim reader (xlm-roberta-base)

Reads one point from a Steam review and says which subject it is about, whether
it is praise or a complaint, and how sure it is. Below a calibrated threshold it
says nothing, and that is a supported answer rather than a failure.

## Measured

- Accuracy 0.476, macro F1 0.385
- Polarity macro F1 0.700
- Calibration error 0.058
- Below 0.61 confidence it says nothing, which leaves it answering 27% of claims at 0.747 accuracy, somewhere in [0.696, 0.792] over 316 claims
- Area under the risk-coverage curve 0.351 (lower is better; it says whether the model knows when it does not know)
- Trained on 4566 claims, validated on 1937, measured on 1177
- Data fingerprint `3b96f2b5bdce045e`, code `ea5cee240b1a`

**Every figure above is from the frozen games**, which the model never saw and
which chose nothing about it, not even the threshold. At that same threshold the
validation games report 0.757, which is
what it scores on games used to build it rather than what a new game gets.
Weakest subjects here: `accessibility` 0.00, `community` 0.00, `content` 0.12, `bugs` 0.25, `atmosphere` 0.28.

## Honest limits

The labels were produced by a language model working from a written category
sheet, not by human adjudication. That makes this a silver standard: agreement
with a model rather than correctness. Two models can agree and be wrong together,
most easily on sarcasm and on the boundaries between categories.

## Licence

Apache-2.0, as is the encoder it was fine-tuned from.
