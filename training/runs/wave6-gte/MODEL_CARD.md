# Claim reader (Alibaba-NLP/gte-multilingual-base)

Reads one point from a Steam review and says which subject it is about, whether
it is praise or a complaint, and how sure it is. Below a calibrated threshold it
says nothing, and that is a supported answer rather than a failure.

## Measured

- Accuracy 0.601, macro F1 0.487
- Polarity macro F1 0.766
- Calibration error 0.109
- Below 0.77 confidence it says nothing, which leaves it answering 47% of claims at 0.798 accuracy, somewhere in [0.775, 0.819] over 1315 claims
- Area under the risk-coverage curve 0.210 (lower is better; it says whether the model knows when it does not know)
- Trained on 8751 claims, validated on 2615, measured on 2797
- Data fingerprint `4bdcc46cd2cdc5df`, code `cb1694554010`

**Every figure above is from the frozen games**, which the model never saw and
which chose nothing about it, not even the threshold. At that same threshold the
validation games report 0.755, which is
what it scores on games used to build it rather than what a new game gets.
Weakest subjects here: `accessibility` 0.00, `licensing` 0.00, `community` 0.21, `policy` 0.29, `tutorial` 0.33.

## Honest limits

The labels were produced by a language model working from a written category
sheet, not by human adjudication. That makes this a silver standard: agreement
with a model rather than correctness. Two models can agree and be wrong together,
most easily on sarcasm and on the boundaries between categories.

## Licence

Apache-2.0, as is the encoder it was fine-tuned from.
