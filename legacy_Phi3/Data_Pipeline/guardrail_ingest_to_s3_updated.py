# dags/guardrail_ingest_to_s3_updated.py
#
# ETL pipeline to ingest Hugging Face datasets to S3 with preprocessing.
# The DAG creates an E -> T -> P -> L flow for each dataset.
# E: Extract, T: Transform, P: Preprocess, L: Load

from __future__ import annotations

from datetime import datetime, timedelta
import os
import logging
import re
import json
from typing import Dict, Any, List, Optional

import boto3
from botocore.exceptions import NoCredentialsError, ClientError

# Core Airflow and provider imports
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
from airflow.hooks.base import BaseHook

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
S3_BUCKET = Variable.get("S3_BUCKET", default_var="guardrail-group-bucket")
S3_PREFIX = Variable.get("S3_PREFIX", default_var="Preprocessed-Fresh/")
MAX_ROWS = int(os.getenv("GUARDRAIL_MAX_ROWS", "25000"))
LOCAL_DATA_DIR = "/opt/airflow/data"

DATASETS: List[Dict[str, Any]] = [
    {
        "name": "safety_prompt",
        "hf": {"path": "PKU-Alignment/BeaverTails", "split": "330k_train"},
        "select": ["prompt", "response", "category", "is_safe"],
    },
    {
        "name": "hate_speech",
        "hf": {"path": "ucberkeley-dlab/measuring-hate-speech", "split": "train"},
        "select": [
            "text",
            "insult",
            "humiliate",
            "dehumanize",
            "violence",
            "genocide",
            "sentiment",
            "respect",
        ],
    },
    {
        "name": "truthful_qa_generation",
        "hf": {"path": "truthful_qa", "name": "generation", "split": "validation"},
        "select": [
            "question",
            "best_answer",
            "correct_answers",
            "incorrect_answers",
            "source",
        ],
    },
    {
        "name": "dolly_instructions",
        "hf": {"path": "databricks/databricks-dolly-15k", "split": "train"},
        "select": ["instruction", "context", "response", "category"],
    },
    {
        "name": "squad_qa",
        "hf": {"path": "squad", "name": "plain_text", "split": "train"},
        "select": ["context", "question", "answers"],
    },
]

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "depends_on_past": False,
}

# ---------------------------------------------------------------------
# Preprocessing Functions
# ---------------------------------------------------------------------


def preprocess_safety_prompt(df):
    """Safety Prompt: lowercase, remove punctuation, parse category JSON to binary flags."""
    import pandas as pd

    logging.info("[PREPROCESS] Processing safety_prompt")

    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower().strip()
        text = re.sub(r'[,;."\']', "", text)
        return text

    df["prompt"] = df["prompt"].apply(clean_text)
    df["response"] = df["response"].apply(clean_text)

    def parse_category(cat_json):
        if pd.isna(cat_json):
            return {
                "animal_abuse": 0,
                "child_abuse": 0,
                "drug_use": 0,
                "financial_crime": 0,
                "hate_speech": 0,
                "self_harm": 0,
                "terrorism": 0,
                "violence": 0,
            }

        if isinstance(cat_json, str):
            try:
                cat_dict = json.loads(cat_json)
            except Exception:
                cat_dict = {}
        elif isinstance(cat_json, dict):
            cat_dict = cat_json
        else:
            cat_dict = {}

        return {
            "animal_abuse": 1 if cat_dict.get("animal_abuse", False) else 0,
            "child_abuse": 1 if cat_dict.get("child_abuse", False) else 0,
            "drug_use": 1 if cat_dict.get("drug_use", False) else 0,
            "financial_crime": 1 if cat_dict.get("financial_crime", False) else 0,
            "hate_speech": 1 if cat_dict.get("hate_speech", False) else 0,
            "self_harm": 1 if cat_dict.get("self_harm", False) else 0,
            "terrorism": 1 if cat_dict.get("terrorism", False) else 0,
            "violence": 1 if cat_dict.get("violence", False) else 0,
        }

    category_df = df["category"].apply(parse_category).apply(pd.Series)
    df = pd.concat([df[["prompt", "response"]], category_df], axis=1)

    if "is_safe" in df.columns:
        df["is_safe"] = df["is_safe"].apply(lambda x: 1 if x else 0)

    logging.info(f"[PREPROCESS] Safety prompt: {len(df)} rows, {len(df.columns)} columns")
    return df


def preprocess_hate_speech(df):
    """Hate Speech: clean text, remove URLs, calculate toxicity, z-score normalization."""
    import pandas as pd
    import numpy as np

    logging.info("[PREPROCESS] Processing hate_speech")

    def clean_text_full(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"\[.*?\]|\(.*?\)", "", text)
        text = re.sub(r"\b(com|uk|www|http|https)\b", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def letters_only(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    df["text"] = df["text"].apply(clean_text_full)
    df["text_letters_only"] = df["text"].apply(letters_only)

    toxicity_cols = ["insult", "humiliate", "dehumanize", "violence", "genocide"]
    for col in toxicity_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["toxicity_mean"] = df[toxicity_cols].mean(axis=1)
    df["label"] = (df["toxicity_mean"] >= 2).astype(int)

    numeric_cols = ["sentiment", "respect"] + toxicity_cols
    for col in numeric_cols:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[f"{col}_z"] = (df[col] - mean) / std
            else:
                df[f"{col}_z"] = 0

    logging.info(f"[PREPROCESS] Hate speech: {len(df)} rows, {len(df.columns)} columns")
    return df


def preprocess_dolly_instructions(df):
    """Dolly: remove quotes, normalize category, create text field, fill empty context."""
    import pandas as pd

    logging.info("[PREPROCESS] Processing dolly_instructions")

    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text)
        text = text.replace('"', "").replace("'", "")
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def normalize_category(cat):
        if pd.isna(cat):
            return "unknown"
        cat = str(cat).lower().strip()
        cat = re.sub(r"[^a-z0-9]+", "_", cat)
        cat = re.sub(r"_+", "_", cat)
        return cat.strip("_")

    df["instruction"] = df["instruction"].apply(clean_text)
    df["context"] = df["context"].apply(clean_text)
    df["response"] = df["response"].apply(clean_text)

    if "category" in df.columns:
        df["category"] = df["category"].apply(normalize_category)

    df["context"] = df["context"].replace("", "N/A")
    df.loc[df["context"].str.strip() == "", "context"] = "N/A"

    df["text"] = df["instruction"] + " " + df["context"]
    df["text"] = df["text"].apply(lambda x: re.sub(r"\s+", " ", x).strip())

    logging.info(f"[PREPROCESS] Dolly: {len(df)} rows, {len(df.columns)} columns")
    return df


def preprocess_truthful_qa(df):
    """Truthful QA: lowercase, flatten answers, create binary labels."""
    import pandas as pd

    logging.info("[PREPROCESS] Processing truthful_qa_generation")
    
    initial_rows = len(df)
    questions = []
    candidate_answers = []
    labels = []
    answer_types = []
    sources = []

    for _, row in df.iterrows():
        question = str(row.get("question", "")).lower()
        source = str(row.get("source", "")).lower()

        # Best answer
        best_answer = row.get("best_answer", "")
        if best_answer and str(best_answer).strip():
            questions.append(question)
            candidate_answers.append(str(best_answer).lower())
            labels.append(1)
            answer_types.append("best")
            sources.append(source)

        # Correct answers
        correct_ans = row.get("correct_answers", [])
        if isinstance(correct_ans, str):
            try:
                correct_ans = json.loads(correct_ans)
            except Exception:
                correct_ans = []
        elif not isinstance(correct_ans, list):
            correct_ans = []

        for ans in correct_ans:
            if ans and str(ans).strip():
                questions.append(question)
                candidate_answers.append(str(ans).lower())
                labels.append(1)
                answer_types.append("correct")
                sources.append(source)

        # Incorrect answers
        incorrect_ans = row.get("incorrect_answers", [])
        if isinstance(incorrect_ans, str):
            try:
                incorrect_ans = json.loads(incorrect_ans)
            except Exception:
                incorrect_ans = []
        elif not isinstance(incorrect_ans, list):
            incorrect_ans = []

        for ans in incorrect_ans:
            if ans and str(ans).strip():
                questions.append(question)
                candidate_answers.append(str(ans).lower())
                labels.append(0)
                answer_types.append("incorrect")
                sources.append(source)

    df_processed = pd.DataFrame(
        {
            "question": questions,
            "candidate_answer": candidate_answers,
            "label": labels,
            "answer_type": answer_types,
            "source": sources,
        }
    )

    logging.info(
        f"[PREPROCESS] Truthful QA: {len(df_processed)} rows (from {initial_rows} original)"
    )
    return df_processed


def preprocess_squad_qa(df):
    """Squad QA: clean text, keep only letters/apostrophes/spaces, extract answer."""
    import pandas as pd

    logging.info("[PREPROCESS] Processing squad_qa")

    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"[^a-z'\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    df["context"] = df["context"].apply(clean_text)
    df["question"] = df["question"].apply(clean_text)

    def extract_answer(answers):
        if pd.isna(answers):
            return ""

        if isinstance(answers, str):
            try:
                answers = json.loads(answers)
            except Exception:
                return ""

        if isinstance(answers, dict):
            text_list = answers.get("text", [])
            if isinstance(text_list, list) and len(text_list) > 0:
                answer = str(text_list[0])
            else:
                answer = ""
        else:
            answer = ""

        answer = str(answer).lower()
        answer = re.sub(r"\s+", " ", answer)
        return answer.strip()

    df["answer"] = df["answers"].apply(extract_answer)
    df = df[["context", "question", "answer"]]
    df = df[df["answer"].str.strip() != ""]

    logging.info(f"[PREPROCESS] Squad QA: {len(df)} rows, {len(df.columns)} columns")
    return df


# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------


def _get_s3_client():
    """
    Create an S3 client using credentials from Airflow connection.
    Uses path-style addressing to avoid SSL issues.
    """
    # Get connection details
    conn = BaseHook.get_connection("aws_guardrail")
    
    # Extract credentials
    aws_access_key_id = conn.login
    aws_secret_access_key = conn.password
    
    # Get region from extra or use default
    extra = conn.extra_dejson if conn.extra else {}
    region_name = extra.get("region_name", "us-east-1")
    
    logging.info(f"Creating S3 client for region: {region_name}")
    
    # Create S3 client with path-style addressing
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
        config=boto3.session.Config(
            s3={'addressing_style': 'path'},  # Use s3.amazonaws.com/bucket/key format
            signature_version='s3v4'
        )
    )
    
    return s3_client


def _upload_to_s3_boto_robust(local_path: str, dataset_name: str) -> str:
    """
    Robustly uploads a local file to S3 using boto3 with a TransferConfig
    to handle large files and prevent timeouts.
    """
    from boto3.s3.transfer import TransferConfig

    config = TransferConfig(
        multipart_threshold=1024 * 1024 * 100,  # 100MB
        max_concurrency=10,
        multipart_chunksize=1024 * 1024 * 25,  # 25MB
        use_threads=True,
    )

    key = (
        f"{S3_PREFIX}{dataset_name}/"
        f"{datetime.utcnow().strftime('%Y/%m/%d')}/"
        f"{dataset_name}.parquet"
    )

    logging.info(f"Uploading to s3://{S3_BUCKET}/{key}")

    try:
        s3_client = _get_s3_client()
        
        s3_client.upload_file(
            Filename=local_path,
            Bucket=S3_BUCKET,
            Key=key,
            Config=config,
        )
    except NoCredentialsError as e:
        raise RuntimeError(
            "AWS credentials not found. Check Airflow connection 'aws_guardrail'."
        ) from e
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        raise RuntimeError(
            f"S3 upload failed with error code {error_code}: {str(e)}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Failed to upload {dataset_name} to S3: {str(e)}"
        ) from e

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

    logging.info(f"[EXTRACT] Dataset '{name}' with args: {hf_args}")
    ds = load_dataset(**hf_args)
    df = pd.DataFrame(ds)
    
    logging.info(f"[EXTRACT] Loaded {len(df)} rows, {len(df.columns)} columns")
    df.to_parquet(local_path, index=False)
    logging.info(f"[EXTRACT] Saved raw data to {local_path}")

    return local_path


def transform_data(
    local_data_path: str, wanted_cols: Optional[List], dataset_name: str
) -> str:
    """
    TRANSFORM: Reads raw local data, cleans and trims it, and saves it as a new file.
    Returns the path to the transformed file.
    """
    import pandas as pd

    logging.info(f"[TRANSFORM] Processing {local_data_path}")
    df = pd.read_parquet(local_data_path)
    initial_rows = len(df)

    if wanted_cols:
        keep = [c for c in wanted_cols if c in df.columns]
        if keep:
            df = df[keep]
            logging.info(f"[TRANSFORM] Selected columns: {keep}")
    
    df = df.dropna(axis=1, how="all")

    def _safe_serialize(x):
        try:
            if isinstance(x, (dict, list)):
                return json.dumps(x, ensure_ascii=False)
            return x
        except Exception:
            return str(x)

    for col in df.select_dtypes(include=["object"]).columns:
        if not df[col].empty and len(df[col]) > 0:
            first_val = df[col].iloc[0]
            if not isinstance(first_val, str) and first_val is not None:
                df[col] = df[col].map(_safe_serialize)

    if len(df) > MAX_ROWS:
        logging.info(f"[TRANSFORM] Sampling from {len(df)} to {MAX_ROWS} rows")
        df = df.sample(n=MAX_ROWS, random_state=42)

    transformed_path = f"{LOCAL_DATA_DIR}/{dataset_name}_transformed.parquet"
    df.to_parquet(transformed_path, index=False)
    logging.info(f"[TRANSFORM] {initial_rows} → {len(df)} rows. Saved to {transformed_path}")

    return transformed_path


def preprocess_data(local_data_path: str, dataset_name: str) -> str:
    """
    PREPROCESS: Apply dataset-specific preprocessing.
    Returns the path to the preprocessed file.
    """
    import pandas as pd

    logging.info(f"[PREPROCESS] Starting for {dataset_name}")
    df = pd.read_parquet(local_data_path)
    initial_rows = len(df)

    # Apply dataset-specific preprocessing
    try:
        if dataset_name == "safety_prompt":
            df = preprocess_safety_prompt(df)
        elif dataset_name == "hate_speech":
            df = preprocess_hate_speech(df)
        elif dataset_name == "dolly_instructions":
            df = preprocess_dolly_instructions(df)
        elif dataset_name == "truthful_qa_generation":
            df = preprocess_truthful_qa(df)
        elif dataset_name == "squad_qa":
            df = preprocess_squad_qa(df)
        else:
            logging.warning(f"[PREPROCESS] No specific preprocessing for {dataset_name}")
    except Exception as e:
        logging.error(f"[PREPROCESS] Error in {dataset_name}: {str(e)}")
        raise

    # Remove duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        logging.info(f"[PREPROCESS] Removed {duplicates} duplicates")

    df = df.reset_index(drop=True)

    preprocessed_path = f"{LOCAL_DATA_DIR}/{dataset_name}_preprocessed.parquet"
    df.to_parquet(preprocessed_path, index=False)

    logging.info(f"[PREPROCESS] Complete: {initial_rows} → {len(df)} rows")
    logging.info(f"[PREPROCESS] Columns: {list(df.columns)}")
    logging.info(f"[PREPROCESS] Saved to {preprocessed_path}")

    return preprocessed_path


def load_data(local_parquet_path: str, dataset_name: str):
    """
    LOAD: Uploads the final preprocessed Parquet file to S3 using the robust uploader.
    """
    logging.info(f"[LOAD] Uploading {dataset_name} to S3")
    
    try:
        _upload_to_s3_boto_robust(local_parquet_path, dataset_name)
    except Exception as e:
        logging.error(f"[LOAD] Failed to upload {dataset_name}: {str(e)}")
        raise

    # Cleanup
    for suffix in ["_raw.parquet", "_transformed.parquet", "_preprocessed.parquet"]:
        temp_path = f"{LOCAL_DATA_DIR}/{dataset_name}{suffix}"
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logging.info(f"[LOAD] Cleaned up {temp_path}")
    
    logging.info(f"[LOAD] ✅ Complete for {dataset_name}")


# ---------------------------------------------------------------------
# DAG Definition
# ---------------------------------------------------------------------
with DAG(
    dag_id="guardrail_ingest_to_s3_updated",
    description="ETL pipeline to ingest Hugging Face datasets to S3 with preprocessing",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    default_args=default_args,
    tags=["s3", "guardrail", "ingest", "etl", "preprocessing"],
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
                    "dataset_name": dataset_name,
                },
            )

            preprocess_task = PythonOperator(
                task_id="preprocess",
                python_callable=preprocess_data,
                op_kwargs={
                    "local_data_path": transform_task.output,
                    "dataset_name": dataset_name,
                },
            )

            load_task = PythonOperator(
                task_id="load",
                python_callable=load_data,
                op_kwargs={
                    "local_parquet_path": preprocess_task.output,
                    "dataset_name": dataset_name,
                },
            )

            # Flow: E -> T -> P -> L
            extract_task >> transform_task >> preprocess_task >> load_task
