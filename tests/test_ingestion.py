import pandas as pd
from src.data_ingestion import ingest_data


def test_ingestion_loads_data():
    sales_path = "data/raw/restaurant_sales_data.csv"
    social_path = "data/raw/social_media_posts.csv"

    sales_df, social_df = ingest_data(sales_path, social_path)

    assert isinstance(sales_df, pd.DataFrame)
    assert isinstance(social_df, pd.DataFrame)

    assert len(sales_df) > 0
    assert len(social_df) > 0


def test_ingestion_columns_exist():
    sales_path = "data/raw/restaurant_sales_data.csv"
    social_path = "data/raw/social_media_posts.csv"

    sales_df, social_df = ingest_data(sales_path, social_path)

    expected_sales_cols = ["date", "restaurant_id", "menu_item_name"]
    expected_social_cols = ["date", "post_text"]

    for col in expected_sales_cols:
        assert col in sales_df.columns

    for col in expected_social_cols:
        assert col in social_df.columns
