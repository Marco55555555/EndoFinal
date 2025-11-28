# ============================================
# data_transformation.py
# Transformación para restaurant_sales_data.csv
# ============================================

import pandas as pd

def transform_sales_data(df):
    """
    Transforma los datos de ventas del restaurante para análisis:
    - Agrupación diaria
    - Ventas totales
    - Sentimiento promedio
    - Número de promociones
    - Clima más común
    """

    # -------------------------------
    # 1. Convertir fecha
    # -------------------------------
    df["date"] = pd.to_datetime(df["date"])

    # -------------------------------
    # 2. Agrupar por fecha (daily)
    # -------------------------------
    daily_df = df.groupby("date").agg({

        # Suma de ventas del día
        "sales": "sum",

        # Sentimiento promedio diario
        "sentiment": "mean",

        # Sumar cuántas promociones hubo en el día
        "has_promotion": "sum",

        # Clima más frecuente
        "weather_condition": lambda x: x.mode()[0] if not x.mode().empty else "Unknown"

    }).reset_index()

    # -------------------------------
    # 3. Renombrar columnas finales
    # -------------------------------
    daily_df.rename(columns={
        "sales": "daily_sales",
        "sentiment": "avg_sentiment",
        "has_promotion": "total_promotions",
    }, inplace=True)

    return daily_df


# ===========================================================
# Función opcional:
# Procesar archivo directamente
# (si deseas usar este módulo sin Streamlit)
# ===========================================================

def load_and_transform(file_path):
    """
    Carga un CSV y devuelve el dataframe transformado.
    Ideal para notebooks y scripts externos.
    """
    df = pd.read_csv(file_path)
    return transform_sales_data(df)


# ===========================================================
# Ejecución directa (solo si llamas el archivo desde terminal)
# python data_transformation.py
# ===========================================================
if __name__ == "__main__":
    try:
        path = "restaurant_sales_data.csv"
        print(f"Cargando archivo: {path}")
        df = pd.read_csv(path)

        transformed = transform_sales_data(df)
        print("Transformación completa:")
        print(transformed.head())

    except Exception as e:
        print("Error:", e)
