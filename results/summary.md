# Results summary

All accuracies are on the same stratified hold-out validation set of 4,200 images.
The 95% uncertainty of one accuracy value is about ±0.3 points (about ±13 images).

| Model | Parameters | Validation accuracy | Errors (of 4,200) |
|---|---|---|---|
| MLP baseline | 235,146 | 0.9786 | 90 |
| SimpleCNN (2 conv blocks) | 421,642 | 0.9895 | 44 |
| SimpleCNN-GAP (control) | 19,466 | 0.9436 | 237 |
| SimpleCNN-64 (size-matched control) | 220,234 | 0.9910 | 38 |
| B: + BatchNorm + dropout | 421,738 | 0.9929 | 30 |
| C: B + augmentation | 421,738 | 0.9926 | 31 |
| D: C + cosine schedule | 421,738 | 0.9933 | 28 |

## Key observations

- The CNN halves the MLP's errors (90 to 44). The size-matched control (fewer parameters than the MLP)
  makes 38 errors, so the gain comes from the convolutions, not from model size.
- The global-average-pooling control underfits (0.9436) because averaging each feature map discards
  where patterns appear in the image.
- BatchNorm and dropout remove the plain CNN's overfitting (44 to 30 errors). B, C and D are within
  the uncertainty of each other.
- Model D is used for the final submission because it is in the best state at its last epoch, which
  matters when the final models are trained on all data without a validation set.
- Manual review of D's 28 validation errors (`misclassified_review.csv`): 4 are mislabelled in the
  dataset, 7 genuinely ambiguous, 4 are 0 versus 6/9 confusions, 9 sloppy but readable and 4 clear
  model mistakes.

## Kaggle

Public score: **0.99232** (username estellearnander03), ensemble of 5 copies of model D trained on all 42,000 images.
