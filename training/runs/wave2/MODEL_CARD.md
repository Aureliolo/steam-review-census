# Claim reader (xlm-roberta-base)

Reads one point from a Steam review and says which subject it is about, whether
it is praise or a complaint, and how sure it is. Below a calibrated threshold it
says nothing, and that is a supported answer rather than a failure.

## Measured

- Accuracy 0.318, macro F1 0.269
- Polarity macro F1 0.638
- Calibration error 0.066
- Below 0.42 confidence it says nothing, which leaves it answering 13% of claims at 0.620 accuracy
- Area under the risk-coverage curve 0.543 (lower is better; it says whether the model knows when it does not know)
- Trained on 3047 claims, validated on 859, measured on 1668
- Data fingerprint `49697fd4fb0f1502`, code `253aebc599c5`

**Every figure above is from the frozen games**, which the model never saw and
which chose nothing about it, not even the threshold. At that same threshold the
validation games report 0.756, which is
what it scores on games used to build it rather than what a new game gets.
Weakest subjects here: `atmosphere` 0.00, `community` 0.00, `licensing` 0.00, `policy` 0.05, `compatibility` 0.10.

## Honest limits

The labels were produced by a language model working from a written category
sheet, not by human adjudication. That makes this a silver standard: agreement
with a model rather than correctness. Two models can agree and be wrong together,
most easily on sarcasm and on the boundaries between categories.

## Licence

Apache-2.0, as is the encoder it was fine-tuned from.
