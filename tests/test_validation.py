import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import logging
from src.data_validation import validate_data

def test_validation_detects_nulls(caplog):
    caplog.set_level(logging.WARNING)

    df = pd.DataFrame({"a": [1, None]})
    validate_data(df)

    assert "valores nulos" in caplog.text.lower()


def test_validation_passes_without_nulls(caplog):
    caplog.set_level(logging.WARNING)

    df = pd.DataFrame({"a": [1, 2]})
    validate_data(df)

    assert "warning" not in caplog.text.lower()
