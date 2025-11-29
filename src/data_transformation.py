import pandas as pd
import logging
from textblob import TextBlob

logging.basicConfig(level=logging.INFO)

# Función para calcular sentimiento
def compute_sentiment(text):
    if pd.isna(text) or not isinstance(text, str):
        return 0.0
    try:
        return TextBlob(str(text)).sentiment.polarity
    except:
        return 0.0


# Transformación principal
def transform_data(sales_df, social_df):

    logging.info("=== Iniciando transformación de datos ===")
    logging.info(f"Filas ventas: {len(sales_df)}, Filas social: {len(social_df)}")

    # 1. Convertir fecha
    sales_df["date"] = pd.to_datetime(sales_df["date"], errors="coerce")
    social_df["date"] = pd.to_datetime(social_df["date"], errors="coerce")

    sales_df = sales_df.dropna(subset=["date"])
    social_df = social_df.dropna(subset=["date"])

    # 2. Crear columna SALES (no existe en tu CSV original)
    logging.info("Calculando columna SALES = precio * cantidad...")
    sales_df["sales"] = (
        sales_df["actual_selling_price"] * sales_df["quantity_sold"]
    )

    # 3. Calcular sentimiento por post
    logging.info("Calculando sentimiento de posts...")
    social_df["post_sentiment"] = social_df["post_text"].apply(compute_sentiment)

    daily_social = social_df.groupby("date").agg({
        "post_sentiment": "mean",
        "post_text": "count"
    }).reset_index()

    daily_social.rename(columns={
        "post_sentiment": "avg_post_sentiment",
        "post_text": "num_posts"
    }, inplace=True)

    # 4. Agrupación diaria de ventas
    logging.info("Agregando datos de ventas...")
    daily_sales = sales_df.groupby("date").agg({
        "sales": "sum",
        "has_promotion": "sum",
        "weather_condition": lambda x: x.mode()[0] if len(x.mode()) else "Unknown"
    }).reset_index()

    daily_sales.rename(columns={
        "sales": "daily_sales",
        "has_promotion": "total_promotions"
    }, inplace=True)

    # 5. Merge ventas + social media
    logging.info("Uniendo ventas + sentimiento social...")
    merged = daily_sales.merge(daily_social, on="date", how="left")

    merged["avg_post_sentiment"] = merged["avg_post_sentiment"].fillna(0)
    merged["num_posts"] = merged["num_posts"].fillna(0)

    logging.info(f"Transformación terminada. Filas finales: {len(merged)}")
    logging.info("=== Fin transformación ===")

    return merged
