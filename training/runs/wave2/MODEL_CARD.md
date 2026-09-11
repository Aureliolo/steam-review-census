# Claim reader (xlm-roberta-base)

Reads one point from a Steam review and says which subject it is about, whether
it is praise or a complaint, and how sure it is. Below a calibrated threshold it
says nothing, and that is a supported answer rather than a failure.

## Measured

- Accuracy 0.392, macro F1 0.255
- Polarity macro F1 0.588
- Calibration error 0.107
- Below 0.42 confidence it says nothing, which leaves it answering 20% of claims at 0.756 accuracy
- Area under the risk-coverage curve 0.423 (lower is better; it says whether the model knows when it does not know)
- Trained on 3047 claims, validated on 859, held out 1668
- Data fingerprint `49697fd4fb0f1502`, code `2474be2bc806`

Games are split whole, never claims, so these figures are about a game the model
never saw. Weakest subjects here: `accessibility` 0.00, `community` 0.00, `compatibility` 0.00, `language` 0.00, `story` 0.05.

## Honest limits

The labels were produced by a language model working from a written category
sheet, not by human adjudication. That makes this a silver standard: agreement
with a model rather than correctness. Two models can agree and be wrong together,
most easily on sarcasm and on the boundaries between categories.

## Licence

Apache-2.0, as is the encoder it was fine-tuned from.
