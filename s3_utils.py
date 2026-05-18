"""
S3 utilities for the Three-Body Problem pipeline.
Handles upload and download of dataset and result files.
"""

import os
import boto3
from botocore.exceptions import ClientError

BUCKET = os.environ.get('S3_BUCKET', 'three-body-dl-data')

RESULT_FILES = [
    'three_body_dataset.pkl',
    'classification_results.pkl',
    'prediction_results.pkl',
    'breen_results.pkl',
    'equilibrium_discovery_results.pkl',
    'classification_confusion_matrices.png',
    'prediction_training_history.png',
    'prediction_examples.png',
    'breen_training_history.png',
    'breen_prediction_examples.png',
    'lagrange_point_discovery.png',
]


def _client():
    return boto3.client('s3')


def upload(local_path, s3_key=None):
    if s3_key is None:
        s3_key = os.path.basename(local_path)
    _client().upload_file(local_path, BUCKET, s3_key)
    print(f"Uploaded {local_path} → s3://{BUCKET}/{s3_key}")


def download(s3_key, local_path=None):
    if local_path is None:
        local_path = s3_key
    try:
        _client().download_file(BUCKET, s3_key, local_path)
        print(f"Downloaded s3://{BUCKET}/{s3_key} → {local_path}")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        raise


def upload_all_results():
    print("\nUploading results to S3...")
    for f in RESULT_FILES:
        if os.path.exists(f):
            upload(f)
        else:
            print(f"Skipping {f} (not found)")
    print("S3 upload complete.")


def download_dataset():
    """Download dataset from S3 if not present locally."""
    if os.path.exists('three_body_dataset.pkl'):
        print("Dataset already present locally, skipping download.")
        return True
    print("Downloading dataset from S3...")
    found = download('three_body_dataset.pkl')
    if not found:
        print("Dataset not found in S3 — will generate locally.")
    return found
