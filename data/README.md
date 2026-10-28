# Data

The data comes from the Kaggle competition **Digit Recognizer**:
https://www.kaggle.com/competitions/digit-recognizer

The CSV files are not included in this repository or in the submitted zip.

## How to get the data

1. Open the competition link above and sign in to Kaggle.
2. Click **Join Competition** and accept the rules.
3. Go to the **Data** tab and download `train.csv` and `test.csv`.
4. Place both files in this folder:

```
data/
├── README.md
├── train.csv   # 42,000 labelled images: label + 784 pixel columns
└── test.csv    # 28,000 unlabelled images: 784 pixel columns
```

Each row is one 28 x 28 grayscale image, with pixel values from 0 to 255.

## Permitted data

Only the official Kaggle `train.csv` and `test.csv` are used. No external
labelled digits are used, including the original MNIST test set or its
labels, as required by the assignment rules.
