# Claim reader (Alibaba-NLP/gte-multilingual-base)

Reads one point from a Steam review and says which subject it is about, whether
it is praise or a complaint, and how sure it is. Below a calibrated threshold it
says nothing, and that is a supported answer rather than a failure.

## Measured

- Accuracy 0.662, macro F1 0.555
- Polarity macro F1 0.812
- Calibration error 0.204
- Below 0.76 confidence it says nothing, which leaves it answering 77% of claims at 0.751 accuracy, somewhere in [0.735, 0.767] over 2895 claims
- Area under the risk-coverage curve 0.160 (lower is better; it says whether the model knows when it does not know)
- Trained on 12523 claims, validated on 2615, measured on 3769
- Data fingerprint `8d9da61190f1ccb1`, code `c1e0d8ec53d4`

**Every figure above is from the frozen games**, which the model never saw and
which chose nothing about it, not even the threshold. At that same threshold the
validation games report 0.750, which is
what it scores on games used to build it rather than what a new game gets.
Weakest subjects here: `licensing` 0.08, `accessibility` 0.08, `vr` 0.11, `community` 0.25, `policy` 0.30.

## Honest limits

The labels were produced by a language model working from a written category
sheet, not by human adjudication. That makes this a silver standard: agreement
with a model rather than correctness. Two models can agree and be wrong together,
most easily on sarcasm and on the boundaries between categories.

## Licence

Apache-2.0, as is the encoder it was fine-tuned from.
