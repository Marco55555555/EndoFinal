import pandas as pd
from src.data_transformation import transform_data

def test_transformation_creates_sales_column():
    # DataFrame de ventas de prueba con todas las columnas necesarias
    sales_df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-01"],
        "actual_selling_price": [10, 12],
        "quantity_sold": [5, 3],
        "has_promotion": [1, 0],
        "weather_condition": ["sunny", "sunny"]
    })

    # DataFrame social de prueba
    social_df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-01"],
        "post_text": ["Me encantó la comida!", "Muy bueno el servicio!"]
    })

    transformed = transform_data(sales_df, social_df)

    # Verificar que se haya creado la columna 'daily_sales'
    assert "daily_sales" in transformed.columns
    # Verificar que las ventas se sumen correctamente
    assert transformed["daily_sales"].iloc[0] == 10*5 + 12*3
    # Verificar que total_promotions esté correcta
    assert transformed["total_promotions"].iloc[0] == 1 + 0

def test_sentiment_column_exists():
    sales_df = pd.DataFrame({
        "date": ["2024-01-01"],
        "actual_selling_price": [10],
        "quantity_sold": [5],
        "has_promotion": [1],
        "weather_condition": ["sunny"]
    })

    social_df = pd.DataFrame({
        "date": ["2024-01-01"],
        "post_text": ["La comida estuvo deliciosa"]
    })

    transformed = transform_data(sales_df, social_df)

    # Verificar que exista la columna de sentimiento promedio
    assert "avg_post_sentiment" in transformed.columns
    # Verificar que exista la columna de número de posts
    assert "num_posts" in transformed.columns
    # Verificar que el sentimiento se haya calculado correctamente (no NaN)
    assert transformed["avg_post_sentiment"].iloc[0] != None
