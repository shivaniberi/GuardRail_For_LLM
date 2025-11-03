from datetime import datetime
import os
from airflow import DAG
from airflow.operators.python import PythonOperator

def upload_test():
    import boto3, pathlib
    bucket = os.getenv("S3_BUCKET", "guardrail-group-bucket")
    prefix = os.getenv("S3_PREFIX", "raw/")
    path = pathlib.Path("/opt/airflow/data"); path.mkdir(parents=True, exist_ok=True)
    local = path / "hello.txt"
    local.write_text("hello from airflow!\n")
    key = f"{prefix}smoke/{datetime.utcnow().strftime('%Y/%m/%d')}/hello.txt"
    boto3.client("s3").upload_file(str(local), bucket, key)
    print(f"Uploaded s3://{bucket}/{key}")

with DAG(
    dag_id="s3_smoke_test",
    start_date=datetime(2025,1,1),
    schedule_interval=None,
    catchup=False,
    tags=["s3","smoke"],
) as dag:
    PythonOperator(task_id="upload_test", python_callable=upload_test)
