# Claim reader (xlm-roberta-base)

Reads one point from a Steam review and says which subject it is about, whether
it is praise or a complaint, and how sure it is. Below a calibrated threshold it
says nothing, and that is a supported answer rather than a failure.

## Measured

- Accuracy 0.426, macro F1 0.342
- Polarity macro F1 0.599
- Calibration error 0.047
- Trained on 2725 claims, validated on 876, held out 1668
- Data fingerprint `7deb903e9c17af2d`, code `1818dd9bdf22`

Games are split whole, never claims, so these figures are about a game the model
never saw. Weakest subjects here: `community` 0.00, `performance` 0.00, `policy` 0.00, `bugs` 0.06, `story` 0.13.

## Honest limits

The labels were produced by a language model working from a written category
sheet, not by human adjudication. That makes this a silver standard: agreement
with a model rather than correctness. Two models can agree and be wrong together,
most easily on sarcasm and on the boundaries between categories.

## Licence

Apache-2.0, as is the encoder it was fine-tuned from.
