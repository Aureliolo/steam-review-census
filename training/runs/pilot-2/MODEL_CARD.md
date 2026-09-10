# Claim reader (xlm-roberta-base)

Reads one point from a Steam review and says which subject it is about, whether
it is praise or a complaint, and how sure it is. Below a calibrated threshold it
says nothing, and that is a supported answer rather than a failure.

## Measured

- Accuracy 0.427, macro F1 0.395
- Polarity macro F1 0.613
- Calibration error 0.129
- Trained on 1629 claims, validated on 417, held out 714
- Data fingerprint `15619efd9a045aaf`, code `224e269ba79a`

Games are split whole, never claims, so these figures are about a game the model
never saw. Weakest subjects here: `compatibility` 0.00, `content` 0.07, `updates` 0.19, `price` 0.25, `performance` 0.30.

## Honest limits

The labels were produced by a language model working from a written category
sheet, not by human adjudication. That makes this a silver standard: agreement
with a model rather than correctness. Two models can agree and be wrong together,
most easily on sarcasm and on the boundaries between categories.

## Licence

Apache-2.0, as is the encoder it was fine-tuned from.
