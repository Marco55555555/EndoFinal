import logging
import yaml
from .data_ingestion import ingest_data
from .data_transformation import transform_data
from .data_validation import validate_data

# generacion del.log
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    filename="pipeline_execution.log",
    level="INFO",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    logging.info("Inicio del pipeline Restaurant Sales ")

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

        # Paso 3: Validación
        validate_data(processed_df)

        # Guardar salida
        output_path = config["output"]["processed_data"]
        processed_df.to_csv(output_path, index=False)

        logging.info(f"Datos procesados guardados en {output_path}")
        logging.info("Pipeline ejecutado exitosamente ")

    except Exception as e:
        logging.error(f"Error en la ejecución: {e}", exc_info=True)

if __name__ == "__main__":
    main()
