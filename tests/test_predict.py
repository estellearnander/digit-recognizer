"""The submission file meets the format required by the assignment."""

import numpy as np
import pandas as pd
import pytest

from digit_recognizer import validate_submission, write_submission


def test_valid_submission_passes(tmp_path):
    path = tmp_path / "submission.csv"
    write_submission(np.arange(28000) % 10, path)
    assert validate_submission(path) is True
    assert open(path).readline().strip() == "ImageId,Label"


def test_missing_row_is_rejected(tmp_path):
    path = tmp_path / "submission.csv"
    write_submission(np.arange(27999) % 10, path)
    with pytest.raises(ValueError):
        validate_submission(path)


def test_label_outside_0_to_9_is_rejected(tmp_path):
    path = tmp_path / "submission.csv"
    pd.DataFrame({"ImageId": np.arange(1, 28001), "Label": np.full(28000, 10)}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        validate_submission(path)


def test_wrong_header_is_rejected(tmp_path):
    path = tmp_path / "submission.csv"
    pd.DataFrame({"Id": np.arange(1, 28001), "Label": np.zeros(28000, dtype=int)}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        validate_submission(path)
