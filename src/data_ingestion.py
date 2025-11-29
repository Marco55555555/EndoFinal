import pandas as pd
import logging

def ingest_data(sales_path, social_path):
    logging.info("Cargando datos de ventas y redes sociales...")

    # Cargar datasets
    sales_df = pd.read_csv(sales_path)
    social_df = pd.read_csv(social_path)

    logging.info(f"Datos cargados: ventas={len(sales_df)}, redes={len(social_df)} filas")

    print("Sales DF columnas:", list(sales_df.columns))
    print("Social DF columnas:", list(social_df.columns))

    return sales_df, social_df
