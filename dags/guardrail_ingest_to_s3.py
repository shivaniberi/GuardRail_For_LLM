# dags/guardrail_ingest_to_s3.py
#
# A formal ETL pipeline to ingest Hugging Face datasets to S3.
# The DAG dynamically creates an E -> T -> L flow for each dataset.
# The 'load' step is enhanced to handle large files and prevent timeouts.

# This is the corrected import statement
from __future__ import annotations

from datetime import datetime, timedelta
import os
import logging
from typing import Dict, Any, List, Optional

# Core Airflow and provider imports
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
S3_BUCKET = os.getenv("S3_BUCKET", "guardrail-group-bucket")
S3_PREFIX = os.getenv("S3_PREFIX", "raw/")
MAX_ROWS = int(os.getenv("GUARDRAIL_MAX_ROWS", "25000"))
LOCAL_DATA_DIR = "/opt/airflow/data"

DATASETS: List[Dict[str, Any]] = [
    {"name": "safety_prompt", "hf": {"path": "PKU-Alignment/BeaverTails", "split": "330k_train"}, "select": None},
    {"name": "hate_speech", "hf": {"path": "ucberkeley-dlab/measuring-hate-speech", "split": "train"},
     "select": ["text", "hate_speech_score", "inferiority", "violence", "hatespeech"]},
    {"name": "truthful_qa_generation", "hf": {"path": "truthful_qa", "name": "generation", "split": "validation"},
     "select": ["question"]},
    {"name": "wino_bias", "hf": {"path": "wino_bias", "name": "type1_pro", "split": "test"}, "select": None},
    {"name": "dolly_instructions", "hf": {"path": "databricks/databricks-dolly-15k", "split": "train"},
     "select": ["instruction", "context", "response"]},
    {"name": "squad_qa", "hf": {"path": "squad", "name": "plain_text", "split": "train"},
     "select": ["context", "question", "answers"]},
    {"name": "multi_nli", "hf": {"path": "multi_nli", "split": "train"},
     "select": ["premise", "hypothesis", "label"]},
]

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "depends_on_past": False,
}

# ---------------------------------------------------------------------
# Reusable Helper Functions
# ---------------------------------------------------------------------

def _upload_to_s3_boto_robust(local_path: str, dataset_name: str) -> str:
    """
    Robustly uploads a local file to S3 using boto3 with a TransferConfig
    to handle large files and prevent timeouts.
    """
    import boto3
    from boto3.s3.transfer import TransferConfig

    # Configure the uploader for multipart uploads for files over 100MB
    config = TransferConfig(
        multipart_threshold=1024 * 1024 * 100,  # 100MB
        max_concurrency=10,
        multipart_chunksize=1024 * 1024 * 25,  # 25MB
        use_threads=True,
    )

    s3_client = boto3.client("s3")
    key = f"{S3_PREFIX}{dataset_name}/{datetime.utcnow().strftime('%Y/%m/%d')}/{dataset_name}.parquet"
    
    s3_client.upload_file(
        Filename=local_path,
        Bucket=S3_BUCKET,
        Key=key,
        Config=config
    )
    
    uploaded_path = f"s3://{S3_BUCKET}/{key}"
    logging.info(f"✅ Uploaded to {uploaded_path}")
    return uploaded_path


# ---------------------------------------------------------------------
# ETL Task Callables
# ---------------------------------------------------------------------

def extract_data(cfg: Dict[str, Any]) -> str:
    """
    EXTRACT: Downloads data from Hugging Face and saves it locally as a Parquet file.
    Returns the path to the local file.
    """
    from datasets import load_dataset
    import pandas as pd
    
    name, hf_args = cfg["name"], cfg["hf"]
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    local_path = f"{LOCAL_DATA_DIR}/{name}_raw.parquet"
    
    logging.info(f"[E]xtracting dataset '{name}' with args: {hf_args}")
    ds = load_dataset(**hf_args)
    df = pd.DataFrame(ds)
    df.to_parquet(local_path, index=False)
    logging.info(f"[E]xtracted and saved raw data to {local_path}")
    
    return local_path

def transform_data(local_data_path: str, wanted_cols: Optional[List], dataset_name: str) -> str:
    """
    TRANSFORM: Reads raw local data, cleans and trims it, and saves it as a new file.
    Returns the path to the transformed file.
    """
    import pandas as pd
    import json
    
    logging.info(f"[T]ransforming data from {local_data_path}")
    df = pd.read_parquet(local_data_path)

    if wanted_cols:
        keep = [c for c in wanted_cols if c in df.columns]
        if keep:
            df = df[keep]
    df = df.dropna(axis=1, how="all")

    def _safe_serialize(x):
        try:
            if isinstance(x, (dict, list)):
                return json.dumps(x, ensure_ascii=False)
            return x
        except Exception:
            return str(x)

    for col in df.select_dtypes(include=['object']).columns:
        if not df[col].empty and not isinstance(df[col].iloc[0], str):
            df[col] = df[col].map(_safe_serialize)

    if len(df) > MAX_ROWS:
        logging.info(f"Sampling down from {len(df)} to {MAX_ROWS} rows.")
        df = df.sample(n=MAX_ROWS, random_state=42)
    
    transformed_path = f"{LOCAL_DATA_DIR}/{dataset_name}_transformed.parquet"
    df.to_parquet(transformed_path, index=False)
    logging.info(f"[T]ransformed data and saved to {transformed_path}")
    
    return transformed_path

def load_data(local_parquet_path: str, dataset_name: str):
    """
    LOAD: Uploads the final transformed Parquet file to S3 using the robust uploader.
    """
    logging.info(f"[L]oading {local_parquet_path} to S3 for dataset '{dataset_name}'")
    _upload_to_s3_boto_robust(local_parquet_path, dataset_name)
    os.remove(local_parquet_path)
    raw_path = local_parquet_path.replace("_transformed.parquet", "_raw.parquet")
    if os.path.exists(raw_path):
        os.remove(raw_path)

# ---------------------------------------------------------------------
# DAG Definition
# ---------------------------------------------------------------------
with DAG(
    dag_id="guardrail_ingest_to_s3",
    description="ETL pipeline to ingest Hugging Face datasets to S3",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    default_args=default_args,
    tags=["s3", "guardrail", "ingest", "etl"],
) as dag:
    for dataset_config in DATASETS:
        dataset_name = dataset_config["name"]
        
        with TaskGroup(group_id=f"etl_{dataset_name}") as etl_group:
            extract_task = PythonOperator(
                task_id="extract",
                python_callable=extract_data,
                op_kwargs={"cfg": dataset_config},
            )
            
            transform_task = PythonOperator(
                task_id="transform",
                python_callable=transform_data,
                op_kwargs={
                    "local_data_path": extract_task.output,
                    "wanted_cols": dataset_config.get("select"),
                    "dataset_name": dataset_name
                },
            )

            load_task = PythonOperator(
                task_id="load",
                python_callable=load_data,
                op_kwargs={
                    "local_parquet_path": transform_task.output,
                    "dataset_name": dataset_name,
                },
            )
            
            extract_task >> transform_task >> load_task