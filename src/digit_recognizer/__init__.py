"""digit_recognizer: reusable code for the Kaggle Digit Recognizer assignment."""

from .data import (
    DigitDataset,
    check_data,
    compute_mean_std,
    load_test,
    load_train,
    make_augmentation,
    make_loaders,
    make_test_loader,
    make_train_loader,
    split_train_val,
)
from .models import MLP, ImprovedCNN, SimpleCNN, SimpleCNNGAP
from .plots import plot_confusion_matrix, plot_history, plot_misclassified, plot_model_comparison
from .predict import predict_proba, validate_submission, write_submission
from .train import count_parameters, evaluate, fit, train_final, train_one_epoch
from .utils import get_device, set_seed

__version__ = "0.1.0"
