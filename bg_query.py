from google.cloud import bigquery
from astropy.table import Table
import logging
import re

class Google_Cloud_query:
    def __init__(self, logger: logging.Logger, table_loc: str, project_id: str, output: str):
        self.logger = logger
        self.table_loc = table_loc
        self.project_id = project_id
        self.output = output

    def run_query(self) -> None:
        client = bigquery.Client(project=self.project_id)
        query = f"SELECT * FROM `{self.table_loc}`"
        self.logger.info(f"Running query: {query}")
        query_job = client.query(query).to_dataframe()
        self.logger.info(f"Query completed. Saving results to {self.output}")
        table = Table.from_pandas(query_job)
        table.write(self.output, format='fits', overwrite=True)

class BG_images:
    def __init__(self, logger: logging.Logger, detection_table: str, image_table: str, gaia_id: int, filter:str):
        self.logger = logger
        self.gaia_id = gaia_id
        self.filter = filter
        self.detection_table = detection_table
        self.image_table = image_table
    
    @staticmethod
    def catalogue_to_image_uri(logger: logging.Logger, filename: str) -> str:
        logger.info(f"Converting catalogue filename to image URI: {filename}")
        uri = re.sub(r"_cat\.fits$", ".fits.fz", filename)
        logger.info(f"Converted URI: {uri}")
        return uri
    
    def query_bg_database(self) -> list:
        gaia_id = int(self.gaia_id)
        filt = str(self.filter)
        query = f"""
            SELECT i.FILENAME
            FROM `{self.detection_table}` AS d
            JOIN `{self.image_table}` AS i 
            ON d.IMAGE_ID = i.IMAGE_ID
            WHERE d.SOURCE_ID = @gaia_id"""
        self.logger.info(f"Running query: {query} with gaia_id={gaia_id} and filter={filt}")
        client = bigquery.Client()
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("gaia_id", "INT64", gaia_id),
                bigquery.ScalarQueryParameter("filt", "STRING", filt),
            ]
        )
        query_job = client.query(query, job_config=job_config)
        results = query_job.result()
        self.logger.info("Query completed. Processing results...")
        gcs_files = [self.catalogue_to_image_uri(self.logger, res.FILENAME) for res in results]
        self.logger.info(f"GCS files -> {len(gcs_files)}...")
        return gcs_files