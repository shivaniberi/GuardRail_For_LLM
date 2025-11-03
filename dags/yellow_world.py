from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

def say_yellow():
    print("Yellow World 👋")

with DAG(
    dag_id="yellow_world",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,  # run manually
    catchup=False,
    default_args={
        "owner": "kaushik",
        "retries": 0,
        "retry_delay": timedelta(minutes=1),
    },
    description="A simple DAG that prints 'Yellow World'",
) as dag:
    hello = PythonOperator(
        task_id="print_yellow",
        python_callable=say_yellow,
    )

    hello

