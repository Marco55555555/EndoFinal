import logging
import yaml
import os
from src.data_ingestion import ingest_data
from src.data_transformation import transform_data
from src.data_validation import validate_data

# === LIMPIAR HANDLERS PARA QUE SE GENERE EL LOG ===
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename="pipeline_execution.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Agregar handler para consola para debug
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)

def main():
    logging.info("Inicio del pipeline Restaurant Sales")

    try:
        # Leer configuración
        with open("config/pipeline_config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # Rutas de los datos
        sales_path = config["data_sources"]["sales_data"]["path"]
        social_path = config["data_sources"]["social_data"]["path"]

        # Paso 1: Ingesta
        sales_df, social_df = ingest_data(sales_path, social_path)

        # Paso 2: Transformación
        processed_df = transform_data(sales_df, social_df)

        # Mostrar info de debug
        logging.info("DataFrame final a guardar:")
        logging.info(f"Filas: {len(processed_df)}")
        logging.info(f"Columnas: {list(processed_df.columns)}")
        print(processed_df.head())

        # Paso 3: Validación
        validate_data(processed_df)

        # Asegurarse que la carpeta de salida exista
        output_path = config["output"]["processed_data"]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Guardar salida
        processed_df.to_csv(output_path, index=False)
        logging.info(f"Datos procesados guardados en {output_path}")
        logging.info("Pipeline ejecutado exitosamente")

    except Exception as e:
        logging.error(f"Error en la ejecución: {e}", exc_info=True)
        print(f"Error en la ejecución: {e}")

if __name__ == "__main__":
    main()
