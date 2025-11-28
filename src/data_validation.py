import logging

def validate_data(df):
    logging.info("Validando datos procesados...")

    nulls = df.isnull().sum().sum()

    if nulls > 0:
        logging.warning(f"Existen {nulls} valores nulos en el dataset procesado.")
    else:
        logging.info("Validación completada sin errores.")
