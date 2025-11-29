import pandas as pd
import logging
from src.data_validation import validate_data


def test_validation_detects_nulls(caplog):
    caplog.set_level(logging.WARNING)

    df = pd.DataFrame({"a": [1, None]})
    validate_data(df)

    # Busca el warning correcto en los logs
    assert "valores nulos" in caplog.text.lower()
    assert "warning" in caplog.text.lower()


def test_validation_passes_without_nulls(caplog):
    caplog.set_level(logging.WARNING)

    df = pd.DataFrame({"a": [1, 2]})
    validate_data(df)

    # No deben aparecer warnings
    assert "valores nulos" not in caplog.text.lower()
    assert "warning" not in caplog.text.lower()
