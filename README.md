# Digit Recognizer with PyTorch

Assignment 2, AI, Machine Learning & Analytics, IE University.

## Objective

Classify 28 x 28 grayscale images of handwritten digits (0 to 9) from the Kaggle competition
[Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) with PyTorch. The project compares an MLP
baseline, a convolutional neural network (CNN) and improved CNNs on a hold-out validation set, and submits an
ensemble of the best configuration to Kaggle.

## Author

Estelle Arnander

- **Kaggle username:** estellearnander03
- **Kaggle public score:** 0.99232

## Project structure

```
digit-recognizer/
├── README.md
├── pyproject.toml              # package metadata and dependencies
├── environment.yml             # conda environment
├── memory.pdf                  # report
├── data/
│   └── README.md               # how to download the Kaggle data (CSV files not included)
├── notebooks/
│   └── digit_recognizer.ipynb  # main notebook, executed, imports the package
├── src/digit_recognizer/
│   ├── __init__.py
│   ├── data.py                 # loading, checks, split, Dataset, DataLoaders, augmentation
│   ├── models.py               # MLP, SimpleCNN, control CNNs, ImprovedCNN
│   ├── train.py                # training and evaluation loops
│   ├── predict.py              # predictions and the submission file
│   ├── plots.py                # figures
│   └── utils.py                # seeds and device
├── results/
│   ├── submission.csv          # the file uploaded to Kaggle
│   ├── summary.md              # key numbers and observations
│   ├── model_comparison.csv
│   ├── misclassified_review.csv
│   ├── models/                 # weights of the 5 final models
│   └── *.png                   # figures
└── tests/                      # pytest tests
```

## Setup

1. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   ```
2. Activate it:
   ```bash
   conda activate digit-recognizer
   ```
3. Install the package in editable mode, from the project root:
   ```bash
   python -m pip install -e .
   ```
4. Check that the package works:
   ```bash
   python -c "import digit_recognizer; print('Package works')"
   ```

## Data

Download `train.csv` and `test.csv` from the Kaggle competition and place them in `data/`. Step-by-step
instructions are in [`data/README.md`](data/README.md).

Only the official Kaggle files are used. No external labelled digits are used, including the original MNIST test
set or its labels.

## Running the analysis

1. Open `notebooks/digit_recognizer.ipynb` in Jupyter or PyCharm.
2. Set the notebook's kernel to the `digit-recognizer` environment.
3. Restart the kernel and run all cells from top to bottom. A full run takes about 35 minutes on an Apple M-series
   GPU (it trains 7 models on the training split and 5 final models on all data).
4. Figures, tables, model weights and `results/submission.csv` are written to `results/`.

## Running the tests

```bash
pytest -q
```

The tests check that every model returns logits of shape [B, 10] and has the parameter count stated in the report,
that the data checks, split and Dataset behave as expected, and that the submission validator accepts a correct file
and rejects broken ones.

## Results summary

All validation results use the same stratified hold-out set of 4,200 images (95% uncertainty about ±0.3 points).

| Model | Parameters | Validation accuracy | Errors (of 4,200) |
|---|---|---|---|
| MLP baseline | 235,146 | 0.9786 | 90 |
| SimpleCNN (2 convolutional blocks) | 421,642 | 0.9895 | 44 |
| SimpleCNN-GAP (control) | 19,466 | 0.9436 | 237 |
| SimpleCNN-64 (size-matched control) | 220,234 | 0.9910 | 38 |
| B: + BatchNorm + dropout | 421,738 | 0.9929 | 30 |
| C: B + augmentation | 421,738 | 0.9926 | 31 |
| D: C + cosine learning-rate schedule | 421,738 | 0.9933 | 28 |

- The CNN halves the MLP's errors, and a CNN with fewer parameters than the MLP still does, so the gain comes from
  the convolutions and not from model size.
- BatchNorm and dropout remove the plain CNN's overfitting. B, C and D are within the validation uncertainty.
- The submission is the average of five copies of model D trained on all 42,000 images: **0.99232** on Kaggle.
- A manual review of model D's 28 validation errors found 4 mislabelled images in the dataset and only 4 clear model
  mistakes.

See `memory.pdf` for the full analysis and `results/summary.md` for a short version.

## Limitations

- All comparisons rest on one hold-out split of 4,200 images, so differences under about 13 images are not
  meaningful.
- The augmented models were still learning at epoch 30. Training them longer is the most promising improvement and
  should be tested on validation first.
- The final ensemble cannot be validated directly, because it is trained on all labelled images.
- Mislabelled images were left in the data.

## Reproducibility

Every random step uses a fixed seed, and every model gets its own freshly created DataLoader. Apple's MPS GPU does
not guarantee bit-identical results on other hardware, so the weights of the five final models are saved in
`results/models/`. The submitted predictions can be reproduced exactly from these files.

To reproduce the submitted predictions from the saved weights, without retraining (run from the project root):

```python
from pathlib import Path
from digit_recognizer import (DigitDataset, compute_mean_std, ensemble_from_weights, get_device,
                              load_test, load_train, make_test_loader, write_submission)

train_df = load_train("data/train.csv")
test_df = load_test("data/test.csv")
mean, std = compute_mean_std(train_df.drop(columns="label").values)
test_loader = make_test_loader(DigitDataset(test_df.values, None, mean, std))

probs = ensemble_from_weights(sorted(Path("results/models").glob("model_d_seed*.pt")), test_loader, get_device())
write_submission(probs.argmax(axis=1), "results/submission_reproduced.csv")
```

