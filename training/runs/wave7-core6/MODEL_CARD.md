# Claim reader (Alibaba-NLP/gte-multilingual-base)

Reads one point from a Steam review and says which subject it is about, whether
it is praise or a complaint, and how sure it is. Below a calibrated threshold it
says nothing, and that is a supported answer rather than a failure.

## Measured

- Accuracy 0.605, macro F1 0.504
- Polarity macro F1 0.783
- Calibration error 0.144
- Below 0.75 confidence it says nothing, which leaves it answering 58% of claims at 0.749 accuracy, somewhere in [0.730, 0.767] over 2175 claims
- Area under the risk-coverage curve 0.216 (lower is better; it says whether the model knows when it does not know)
- Trained on 12523 claims, validated on 2615, measured on 3769
- Data fingerprint `f577ab6ad6fcbd46`, code `7af82f558c0a`

**Every figure above is from the frozen games**, which the model never saw and
which chose nothing about it, not even the threshold. At that same threshold the
validation games report 0.751, which is
what it scores on games used to build it rather than what a new game gets.
Weakest subjects here: `licensing` 0.00, `accessibility` 0.08, `community` 0.09, `policy` 0.21, `vr` 0.25.

## Honest limits

The labels were produced by a language model working from a written category
sheet, not by human adjudication. That makes this a silver standard: agreement
with a model rather than correctness. Two models can agree and be wrong together,
most easily on sarcasm and on the boundaries between categories.

## Licence

Apache-2.0, as is the encoder it was fine-tuned from.
