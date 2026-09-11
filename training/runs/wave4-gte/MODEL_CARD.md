# Claim reader (Alibaba-NLP/gte-multilingual-base)

Reads one point from a Steam review and says which subject it is about, whether
it is praise or a complaint, and how sure it is. Below a calibrated threshold it
says nothing, and that is a supported answer rather than a failure.

## Measured

- Accuracy 0.608, macro F1 0.512
- Polarity macro F1 0.783
- Calibration error 0.067
- Below 0.77 confidence it says nothing, which leaves it answering 37% of claims at 0.803 accuracy, somewhere in [0.771, 0.832] over 666 claims
- Area under the risk-coverage curve 0.236 (lower is better; it says whether the model knows when it does not know)
- Trained on 5988 claims, validated on 1937, measured on 1800
- Data fingerprint `8e89a2c271520666`, code `4683c2b2eca3`

**Every figure above is from the frozen games**, which the model never saw and
which chose nothing about it, not even the threshold. At that same threshold the
validation games report 0.755, which is
what it scores on games used to build it rather than what a new game gets.
Weakest subjects here: `accessibility` 0.00, `policy` 0.21, `tutorial` 0.31, `bugs` 0.32, `community` 0.36.

## Honest limits

The labels were produced by a language model working from a written category
sheet, not by human adjudication. That makes this a silver standard: agreement
with a model rather than correctness. Two models can agree and be wrong together,
most easily on sarcasm and on the boundaries between categories.

## Licence

Apache-2.0, as is the encoder it was fine-tuned from.
