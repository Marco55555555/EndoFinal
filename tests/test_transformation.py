import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.data_transformation import transform_data

def test_transformation_creates_sales_column():
    sales_df = pd.DataFrame({
        "date": ["2024-01-01"],
        "actual_selling_price": [10],
        "quantity_sold": [5]
    })

    social_df = pd.DataFrame({
        "date": ["2024-01-01"],
        "post_text": ["excelente comida"]
    })

    merged = transform_data(sales_df, social_df)

    assert "sales" in merged.columns
    assert merged["sales"].iloc[0] == 50  # 10 * 5

def test_sentiment_column_exists():
    sales_df = pd.DataFrame({
        "date": ["2024-01-01"],
        "actual_selling_price": [12],
        "quantity_sold": [3]
    })

    social_df = pd.DataFrame({
        "date": ["2024-01-01"],
        "post_text": ["muy bueno"]
    })

    merged = transform_data(sales_df, social_df)

    assert "sentiment" in merged.columns
