# Claim reader (Alibaba-NLP/gte-multilingual-base)

Reads one point from a Steam review and says which subject it is about, whether
it is praise or a complaint, and how sure it is. Below a calibrated threshold it
says nothing, and that is a supported answer rather than a failure.

## Measured

- Accuracy 0.606, macro F1 0.479
- Polarity macro F1 0.760
- Calibration error 0.082
- Below 0.78 confidence it says nothing, which leaves it answering 43% of claims at 0.808 accuracy, somewhere in [0.783, 0.830] over 1071 claims
- Area under the risk-coverage curve 0.217 (lower is better; it says whether the model knows when it does not know)
- Trained on 7579 claims, validated on 1937, measured on 2495
- Data fingerprint `6dde398537c198d2`, code `69ef2a6aa406`

**Every figure above is from the frozen games**, which the model never saw and
which chose nothing about it, not even the threshold. At that same threshold the
validation games report 0.757, which is
what it scores on games used to build it rather than what a new game gets.
Weakest subjects here: `accessibility` 0.00, `licensing` 0.00, `community` 0.14, `policy` 0.29, `vr` 0.33.

## Honest limits

The labels were produced by a language model working from a written category
sheet, not by human adjudication. That makes this a silver standard: agreement
with a model rather than correctness. Two models can agree and be wrong together,
most easily on sarcasm and on the boundaries between categories.

## Licence

Apache-2.0, as is the encoder it was fine-tuned from.
