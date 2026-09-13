# Claim reader (xlm-roberta-base)

Reads one point from a Steam review and says which subject it is about, whether
it is praise or a complaint, and how sure it is. Below a calibrated threshold it
says nothing, and that is a supported answer rather than a failure.

## Measured

- Accuracy 0.190, macro F1 0.161
- Polarity macro F1 0.186
- Calibration error 0.063
- Trained on 430 claims, validated on 126, held out 125
- Data fingerprint `6bc3a9a40e587b77`, code `4a9e8e58ef58`

Games are split whole, never claims, so these figures are about a game the model
never saw. Weakest subjects here: `audio` 0.00, `genre` 0.00, `language` 0.00, `monetisation` 0.00, `story` 0.00.

## Honest limits

The labels were produced by a language model working from a written category
sheet, not by human adjudication. That makes this a silver standard: agreement
with a model rather than correctness. Two models can agree and be wrong together,
most easily on sarcasm and on the boundaries between categories.

## Licence

Apache-2.0, as is the encoder it was fine-tuned from.
