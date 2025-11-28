import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.data_ingestion import ingest_data

def test_ingestion_loads_data():
    sales_path = "data/raw/restaurant_sales_data.csv"
    social_path = "data/raw/social_media_posts.csv"

    sales_df, social_df = ingest_data(sales_path, social_path)

    assert len(sales_df) > 0
    assert len(social_df) > 0


def test_ingestion_columns_exist():
    sales_path = "data/raw/restaurant_sales_data.csv"
    social_path = "data/raw/social_media_posts.csv"

    sales_df, social_df = ingest_data(sales_path, social_path)

    assert "date" in sales_df.columns
    assert "post_text" in social_df.columns
