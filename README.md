# 🚀 MLOps Churn Prediction Pipeline on AWS

<div align="center">
<img src="https://img.shields.io/badge/AWS-SageMaker-orange" alt="SageMaker">
<img src="https://img.shields.io/badge/Kubernetes-EKS-blue" alt="EKS">
<img src="https://img.shields.io/badge/MLflow-Tracking-red" alt="MLflow">
<img src="https://img.shields.io/badge/XGBoost-Model-green" alt="XGBoost">
<img src="https://img.shields.io/badge/FastAPI-REST-teal" alt="FastAPI">
</div>

## 📋 Project Overview

This project implements a complete end-to-end MLOps pipeline for customer churn prediction on the AWS cloud. It automates the entire machine learning lifecycle, from data ingestion and model training to deployment and monitoring. The architecture is designed to be robust, scalable, and maintainable, leveraging a modern MLOps stack.

***

## 🎯 Key Features

* **Automated ML Pipeline**: A SageMaker Pipeline automatically triggers when new data is uploaded to an S3 bucket.
* **Experiment Tracking**: An MLflow server, running on a Kubernetes (EKS) cluster, tracks all experiments, including parameters, metrics, and model artifacts.
* **Robust Model Training**: An XGBoost model is trained with comprehensive hyperparameter and artifact tracking.
* **Model Governance**: The SageMaker Model Registry is used to version models and includes a manual approval step before deployment.
* **Scalable Deployment**: The approved model is served as a REST API using FastAPI on an EKS cluster, ensuring high availability and scalability.
* **CI/CD Automation**: GitHub Actions are configured to automate the build and deployment processes, triggered by changes in the repository.
* **Web Interface**: A user-friendly dashboard (placeholder for extension) allows for real-time predictions.

***

## 🏛️ Architecture Diagram

```mermaid
graph TD
    subgraph "Data & Trigger"
        A[User uploads data] -->|1. Upload CSV| B(S3 Raw Data Bucket);
        B -->|2. S3 Event Notification| C(λ Lambda Function);
    end

    subgraph "SageMaker ML Pipeline"
        C -->|3. Starts Pipeline| D(SageMaker Pipeline: churn-pipeline);
        D --> D1(Step: Preprocessing);
        D1 --> D2(Step: Training);
        D2 --> D3(Step: Evaluation);
        D3 --> D4(Step: Register Model);
    end

    subgraph "Experiment Tracking"
        D2 -->|Logs Metrics & Artifacts| E(MLflow Server on EKS);
    end

    subgraph "Model Governance"
        D4 -->|If evaluation passes| F{SageMaker Model Registry};
        F -->|Manual Approval| G(Approved Model);
    end

    subgraph "CI/CD & Deployment"
        H(GitHub Repository) -->|On Push to main| I(GitHub Actions);
        I -->|Pulls Approved Model| G;
        I -->|5. Build & Push Docker Image| J(Amazon ECR);
        J -->|6. Deploy to EKS| K(EKS Cluster);
    end

    subgraph "Inference Service"
        K -- hosts --> L(FastAPI Service);
        M[User/Application] -->|7. POST /predict| L;
        L -->|Returns Prediction| M;
    end

    style B fill:#FF9900,stroke:#333,stroke-width:2px
    style C fill:#FF9900,stroke:#333,stroke-width:2px
    style D fill:#22724E,stroke:#333,stroke-width:2px,color:#fff
    style F fill:#22724E,stroke:#333,stroke-width:2px,color:#fff
    style E fill:#db5a3a,stroke:#333,stroke-width:2px,color:#fff
    style K fill:#326CE5,stroke:#333,stroke-width:2px,color:#fff
    style J fill:#FF9900,stroke:#333,stroke-width:2px
    style L fill:#009485,stroke:#333,stroke-width:2px,color:#fff
````

-----

## ✅ Prerequisites

Before you begin, ensure you have the following:

  * An active **AWS Account** with IAM permissions to create the resources mentioned in this guide.
  * **AWS CLI** configured locally. You can configure it by running `aws configure`.
  * **Docker** installed and running on your local machine.
  * The following command-line tools installed:
      * `kubectl` (Kubernetes command-line tool)
      * `helm` (Kubernetes package manager)
      * `eksctl` (Official CLI for Amazon EKS)
  * **Python 3.9** or newer.
  * A **GitHub** account.

-----

## 🏗️ Part 1: Infrastructure Setup

### Step 1: Prepare Churn Dataset and S3 Buckets

First, set up your project directory, download the dataset, and create the S3 buckets that will be used for storing data and model artifacts.

```bash
# Create a project directory and navigate into it
mkdir mlops-churn-project
cd mlops-churn-project

# Download the Telco Customer Churn dataset
wget [https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) -O churn_data.csv

# !!! IMPORTANT: Define a unique identifier for your resources (e.g., your initials and date)
export UNIQUE_ID="[YOUR-UNIQUE-ID]"
export AWS_REGION="ap-south-1" # Or your preferred AWS region

# Create the S3 buckets
aws s3 mb s3://mlops-churn-raw-data-$UNIQUE_ID --region $AWS_REGION
aws s3 mb s3://mlops-churn-processed-data-$UNIQUE_ID --region $AWS_REGION
aws s3 mb s3://mlops-churn-model-artifacts-$UNIQUE_ID --region $AWS_REGION

# Upload the raw dataset to the appropriate S3 bucket
aws s3 cp churn_data.csv s3://mlops-churn-raw-data-$UNIQUE_ID/raw/
```

**Note:** Replace `[YOUR-UNIQUE-ID]` with a unique string to avoid bucket name conflicts.

### Step 2: Create IAM Roles

You need two IAM roles: one for SageMaker to execute pipeline steps and another for the Lambda function to trigger the pipeline.

1.  **SageMaker Execution Role**:

      * Go to the **IAM Console** in AWS.
      * Navigate to **Roles** and click **Create role**.
      * Select **AWS service** for the trusted entity type.
      * Choose **SageMaker** as the use case.
      * Attach the following policies: `AmazonSageMakerFullAccess`, `AmazonS3FullAccess`.
      * Name the role `SageMakerChurnRole` and create it.

2.  **Lambda Execution Role**:

      * In the IAM Console, create another role.
      * Select **AWS service** and choose **Lambda** as the use case.
      * Attach the following policies: `AWSLambdaBasicExecutionRole`, `AmazonS3ReadOnlyAccess`, `AmazonSageMakerFullAccess`.
      * Name the role `LambdaChurnTriggerRole` and create it.

### Step 3: Set Up Amazon SageMaker Studio

SageMaker Studio will be our integrated development environment (IDE) for managing the machine learning pipeline.

  * Navigate to the **Amazon SageMaker** service in the AWS Console.
  * Click on **Studio** from the left-hand menu.
  * Click **Create Domain** and select **Quick setup**.
  * In the **Execution role** dropdown, choose the `SageMakerChurnRole` you created earlier.
  * Complete the setup. This process can take 10-15 minutes.

### Step 4: Create the Lambda Trigger Function

This function will be automatically triggered whenever a new `.csv` file is uploaded to the raw data S3 bucket.

  * Go to the **AWS Lambda Console** and click **Create function**.

  * Select **Author from scratch**.

  * **Function name**: `trigger-churn-pipeline`

  * **Runtime**: `Python 3.9`

  * **Architecture**: `x86_64`

  * **Permissions**: Expand "Change default execution role," select **Use an existing role**, and choose `LambdaChurnTriggerRole`.

  * Click **Create function**.

  * **Add Code**: In the Code source editor, paste the following Python code:

    ```python
    import json
    import boto3
    import os

    PIPELINE_NAME = os.environ['PIPELINE_NAME']

    def lambda_handler(event, context):
        sm_client = boto3.client('sagemaker')
        
        try:
            # Start the SageMaker Pipeline execution
            response = sm_client.start_pipeline_execution(
                PipelineName=PIPELINE_NAME
            )
            print(f"Pipeline execution started: {response['PipelineExecutionArn']}")
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Pipeline triggered successfully',
                    'executionArn': response['PipelineExecutionArn']
                })
            }
        except Exception as e:
            print(f"Error starting pipeline: {str(e)}")
            return {
                'statusCode': 500,
                'body': json.dumps({'message': 'Error triggering pipeline', 'error': str(e)})
            }
    ```

  * **Add Environment Variable**:

      * Go to the **Configuration** tab and then **Environment variables**.
      * Add a variable:
          * **Key**: `PIPELINE_NAME`
          * **Value**: `churn-pipeline`

  * **Add Trigger**:

      * In the function overview, click **Add trigger**.
      * Select **S3** as the source.
      * **Bucket**: Choose `mlops-churn-raw-data-[YOUR-UNIQUE-ID]`.
      * **Event types**: Select **All object create events**.
      * **Prefix**: `raw/`
      * Acknowledge the recursive invocation warning and click **Add**.

### Step 5: Create the EKS Cluster

This Kubernetes cluster will host our MLflow server and the final prediction API.

```bash
# Use eksctl to create the cluster. This can take 15-20 minutes.
eksctl create cluster \
  --name churn-mlops \
  --region $AWS_REGION \
  --nodegroup-name standard-workers \
  --node-type t3.medium \
  --nodes 2 \
  --nodes-min 2 \
  --nodes-max 4 \
  --managed
```

### Step 6: Deploy MLflow on EKS

We will use Helm to deploy a production-ready MLflow instance with a PostgreSQL backend for metadata storage.

```bash
# Add the Bitnami Helm chart repository
helm repo add bitnami [https://charts.bitnami.com/bitnami](https://charts.bitnami.com/bitnami)
helm repo update

# Create a dedicated namespace for MLflow
kubectl create namespace mlflow

# Install MLflow using Helm with a LoadBalancer service
helm install mlflow bitnami/mlflow \
  --namespace mlflow \
  --set service.type=LoadBalancer \
  --set postgresql.enabled=true \
  --set postgresql.auth.username=mlflow \
  --set postgresql.auth.password=mlflow123 \
  --set postgresql.auth.database=mlflow

# Get the external IP for the MLflow UI. Wait for the EXTERNAL-IP to be assigned.
kubectl get svc -n mlflow -w
```

Once an `EXTERNAL-IP` is assigned to the `mlflow` service, you can access your MLflow dashboard at `http://<YOUR-MLFLOW-EXTERNAL-IP>`.

-----

## 🤖 Part 2: Project Files and Scripts

You will need to create a project structure as shown below. The content for each file is provided.

### Project Structure

```
mlops-churn-project/
├── .github/
│   └── workflows/
│       └── main.yml
├── scripts/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── churn_pipeline.py
│   └── requirements.txt
├── api/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements-api.txt
└── k8s/
    ├── deployment.yaml
    └── service.yaml
```

### Script Files (`scripts/`)

#### `scripts/requirements.txt`

```
sagemaker
boto3
scikit-learn
pandas
xgboost
mlflow
```

#### `scripts/preprocessing.py`

```python
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default="/opt/ml/processing/input")
    args = parser.parse_args()

    # Load data from the input path
    input_file = os.path.join(args.input_path, 'raw/churn_data.csv')
    df = pd.read_csv(input_file)

    # Simple data cleaning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop(['customerID'], axis=1, inplace=True)
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    # One-hot encode categorical features
    # In a real-world scenario, you would use a more robust method like ColumnTransformer
    df_processed = pd.get_dummies(df, drop_first=True)

    # Split data
    train, test = train_test_split(df_processed, test_size=0.2, random_state=42)
    
    # Define output paths
    train_path = "/opt/ml/processing/train"
    test_path = "/opt/ml/processing/test"
    
    # Save processed data
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    train.to_csv(os.path.join(train_path, 'train.csv'), index=False)
    test.to_csv(os.path.join(test_path, 'test.csv'), index=False)

    print("Preprocessing complete.")
```

#### `scripts/train.py`

```python
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import argparse
import os
import mlflow
import mlflow.xgboost

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # SageMaker and MLflow arguments
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--mlflow_tracking_uri", type=str)

    args = parser.parse_args()

    # Set up MLflow
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment("churn-prediction-sagemaker")

    with mlflow.start_run():
        # Load data
        train_df = pd.read_csv(os.path.join(args.train, 'train.csv'))
        y_train = train_df['Churn']
        X_train = train_df.drop('Churn', axis=1)

        # Train XGBoost model
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 5,
            'eta': 0.1,
            'gamma': 0.1,
            'subsample': 0.8
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        # Log parameters and metrics
        mlflow.log_params(params)
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        mlflow.log_metric("train_accuracy", accuracy)
        print(f"Train Accuracy: {accuracy}")

        # Log model with MLflow
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name="churn-xgboost-model"
        )
        
        # Save model artifact for SageMaker
        model_path = os.path.join(args.model_dir, "model.xgb")
        model.save_model(model_path)
        print(f"Model saved to {model_path}")
```

#### `scripts/evaluate.py`

```python
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
import os
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="/opt/ml/processing/test")
    parser.add_argument("--model-path", type=str, default="/opt/ml/processing/model")
    parser.add_argument("--output-path", type=str, default="/opt/ml/processing/evaluation")
    args = parser.parse_args()

    # Load data and model
    test_df = pd.read_csv(os.path.join(args.test, 'test.csv'))
    y_test = test_df['Churn']
    X_test = test_df.drop('Churn', axis=1)
    
    model = xgb.Booster()
    model.load_model(os.path.join(args.model_path, 'model.xgb'))
    dtest = xgb.DMatrix(X_test)

    # Evaluate model
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Create evaluation report
    report = {
        'metrics': {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
    }
    
    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, 'evaluation.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    print("Evaluation report generated.")
    print(report)
```

#### `scripts/churn_pipeline.py`

```python
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.step_collections import RegisterModel
import boto3

# --- Configuration ---
# !!! IMPORTANT: Replace with your specific details
UNIQUE_ID = "[YOUR-UNIQUE-ID]"
AWS_ACCOUNT_ID = "[YOUR-AWS-ACCOUNT-ID]"
AWS_REGION = "ap-south-1" # Or your preferred AWS region
MLFLOW_TRACKING_URI = "http://[YOUR-MLFLOW-EXTERNAL-IP]"

ROLE_ARN = f"arn:aws:iam::{AWS_ACCOUNT_ID}:role/SageMakerChurnRole"
PIPELINE_NAME = "churn-pipeline"

# S3 Buckets
raw_data_s3_uri = f"s3://mlops-churn-raw-data-{UNIQUE_ID}"
processed_data_s3_uri = f"s3://mlops-churn-processed-data-{UNIQUE_ID}"
model_artifacts_s3_uri = f"s3://mlops-churn-model-artifacts-{UNIQUE_ID}"

# Get the default SageMaker session
sagemaker_session = sagemaker.Session()
sklearn_image_uri = sagemaker.image_uris.retrieve('sklearn', sagemaker_session.boto_region_name, '0.23-1')

# --- Step 1: Preprocessing ---
script_processor = ScriptProcessor(
    image_uri=sklearn_image_uri,
    command=['python3'],
    instance_type='ml.t3.medium',
    instance_count=1,
    base_job_name='churn-preprocess',
    role=ROLE_ARN,
    sagemaker_session=sagemaker_session
)

step_preprocess = ProcessingStep(
    name="PreprocessChurnData",
    processor=script_processor,
    inputs=[ProcessingInput(source=raw_data_s3_uri, destination='/opt/ml/processing/input')],
    outputs=[
        ProcessingOutput(output_name="train", source='/opt/ml/processing/train', destination=f"{processed_data_s3_uri}/train"),
        ProcessingOutput(output_name="test", source='/opt/ml/processing/test', destination=f"{processed_data_s3_uri}/test")
    ],
    code='scripts/preprocessing.py'
)

# --- Step 2: Training ---
image_uri = sagemaker.image_uris.retrieve(framework='xgboost', region=AWS_REGION, version='1.5-1')

xgb_estimator = Estimator(
    image_uri=image_uri,
    instance_type='ml.m5.large',
    instance_count=1,
    role=ROLE_ARN,
    output_path=f"{model_artifacts_s3_uri}/training-jobs",
    sagemaker_session=sagemaker_session,
    hyperparameters={
        'mlflow_tracking_uri': MLFLOW_TRACKING_URI
    },
    entry_point='train.py',
    source_dir='scripts'
)

step_train = TrainingStep(
    name="TrainXGBoostModel",
    estimator=xgb_estimator,
    inputs={
        "train": TrainingInput(s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri, content_type="text/csv")
    },
)

# --- Step 3: Evaluation ---
step_evaluate = ProcessingStep(
    name="EvaluateModel",
    processor=script_processor, # Re-using the sklearn processor
    inputs=[
        ProcessingInput(source=step_preprocess.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri, destination="/opt/ml/processing/test"),
        ProcessingInput(source=step_train.properties.ModelArtifacts.S3ModelArtifacts, destination="/opt/ml/processing/model")
    ],
    outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation", destination=f"{model_artifacts_s3_uri}/evaluation-report")],
    code='scripts/evaluate.py'
)

# --- Step 4: Register Model ---
step_register = RegisterModel(
    name="RegisterChurnModel",
    estimator=xgb_estimator,
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.t2.medium", "ml.m5.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name="ChurnModelPackageGroup",
    approval_status="PendingManualApproval",
    model_metrics={
        "ModelQuality": {
            "Statistics": {
                "ContentType": "application/json",
                "S3Uri": step_evaluate.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri + "/evaluation.json"
            }
        }
    }
)

# --- Create and Execute Pipeline ---
pipeline = Pipeline(
    name=PIPELINE_NAME,
    parameters=[],
    steps=[step_preprocess, step_train, step_evaluate, step_register]
)

if __name__ == "__main__":
    print(f"Creating/Updating and executing pipeline: {PIPELINE_NAME}")
    pipeline.upsert(role_arn=ROLE_ARN)
    execution = pipeline.start()
    print(f"Pipeline execution started with ARN: {execution.arn}")
```

**Important**: Before running `churn_pipeline.py`, you must replace `[YOUR-UNIQUE-ID]`, `[YOUR-AWS-ACCOUNT-ID]`, and `[YOUR-MLFLOW-EXTERNAL-IP]` with your actual values.

### API and Kubernetes Files

#### `api/requirements-api.txt`

```
fastapi
uvicorn[standard]
pydantic
xgboost
numpy
scikit-learn
```

#### `api/app.py`

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field
import xgboost as xgb
import pandas as pd
import os

app = FastAPI(title="Churn Prediction API")

# This Pydantic model defines the structure of the input data for a single prediction.
# It should match the raw features before preprocessing.
class CustomerFeatures(BaseModel):
    gender: str = "Male"
    SeniorCitizen: int = 0
    Partner: str = "Yes"
    Dependents: str = "No"
    tenure: int = 1
    PhoneService: str = "No"
    MultipleLines: str = "No phone service"
    InternetService: str = "DSL"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "Yes"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 29.85
    TotalCharges: float = 29.85

model = None
model_columns = None

@app.on_event("startup")
def load_model():
    """Load the XGBoost model and expected columns from disk at startup."""
    global model, model_columns
    model_path = os.environ.get("MODEL_PATH", "model.xgb")
    model = xgb.Booster()
    model.load_model(model_path)
    # The columns used for training are needed to ensure prediction data has the same structure.
    # In a real pipeline, these columns would be saved as an artifact during training.
    # For now, we define them here based on the preprocessing script's output.
    global model_columns
    model_columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male', 'Partner_Yes', 
                     'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_No phone service', 'MultipleLines_Yes', 
                     'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No internet service', 
                     'OnlineSecurity_Yes', 'OnlineBackup_No internet service', 'OnlineBackup_Yes', 
                     'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No internet service',
                     'TechSupport_Yes', 'StreamingTV_No internet service', 'StreamingTV_Yes', 
                     'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_One year', 
                     'Contract_Two year', 'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)',
                     'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

@app.get("/")
def read_root():
    return {"message": "Welcome to the Churn Prediction API"}

@app.post("/predict")
def predict_churn(features: CustomerFeatures):
    """Predict churn probability based on customer features."""
    # Convert input data to a pandas DataFrame
    input_df = pd.DataFrame([features.dict()])
    
    # Preprocess the input data to match the training format
    # This uses one-hot encoding similar to the training script.
    processed_df = pd.get_dummies(input_df)
    
    # Align columns with the model's training data
    processed_df = processed_df.reindex(columns=model_columns, fill_value=0)
    
    # Create DMatrix for XGBoost
    dmatrix = xgb.DMatrix(processed_df)
    
    # Make prediction
    probability = model.predict(dmatrix)[0]
    
    return {
        "churn_prediction": "Yes" if probability > 0.5 else "No",
        "churn_probability": float(probability)
    }
```

#### `api/Dockerfile`

```dockerfile
# Use a slim Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the application code and model into the container
# The model.xgb file will be added during the CI/CD process
COPY ./api/app.py /app/
COPY ./api/requirements-api.txt /app/

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements-api.txt

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application using uvicorn
# The model will be downloaded into /app/model.xgb by the CI/CD pipeline
ENV MODEL_PATH=/app/model.xgb
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

#### `k8s/deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: churn-api-deployment
  namespace: default
  labels:
    app: churn-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: churn-api
  template:
    metadata:
      labels:
        app: churn-api
    spec:
      containers:
      - name: churn-api-container
        # !!! IMPORTANT: Replace this with your ECR image URI from the CI/CD pipeline
        image: [YOUR-AWS-ACCOUNT-ID].dkr.ecr.[YOUR-AWS-REGION][.amazonaws.com/churn-prediction-api:latest](https://.amazonaws.com/churn-prediction-api:latest)
        ports:
        - containerPort: 8080
        env:
        - name: MODEL_PATH
          value: "/app/model.xgb"
```

#### `k8s/service.yaml`

```yaml
apiVersion: v1
kind: Service
metadata:
  name: churn-api-service
  namespace: default
spec:
  selector:
    app: churn-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
  type: LoadBalancer
```

-----

## 🚀 Quick Start Guide

1.  **Clone Repository**: `git clone <your-repo-url>`

2.  **Setup Infrastructure**: Follow all steps in **Part 1: Infrastructure Setup**.

3.  **Create Project Files**: Create all the files and directories as described in **Part 2**.

4.  **Trigger the Pipeline**:

      * **Option A (Manual)**: Run the pipeline script locally to create the initial pipeline definition on SageMaker.
        ```bash
        # Ensure you have authenticated with AWS CLI
        pip install -r scripts/requirements.txt
        python scripts/churn_pipeline.py
        ```
      * **Option B (Automatic)**: Upload the `churn_data.csv` file again to trigger the Lambda function.
        ```bash
        aws s3 cp churn_data.csv s3://mlops-churn-raw-data-[YOUR-UNIQUE-ID]/raw/
        ```

5.  **Approve Model in SageMaker**:

      * Navigate to **SageMaker Studio**.
      * From the launcher, select **Model Registry**.
      * Find the `ChurnModelPackageGroup`.
      * Select the latest model version, and click **Update approval status**.
      * Change the status to **Approved**.

6.  **Deploy the API (Manual Steps)**:

      * First, create an **Amazon ECR** repository named `churn-prediction-api`.
      * Download the approved model artifact (`model.tar.gz`) from S3, extract it to get `model.xgb`, and place it in your `api/` directory.
      * Build and push your Docker image to ECR.
      * Update `k8s/deployment.yaml` with your ECR image URI.
      * Apply the Kubernetes manifests:
        ```bash
        # Make sure kubectl is configured to talk to your EKS cluster
        # aws eks --region [YOUR-AWS-REGION] update-kubeconfig --name churn-mlops

        # Apply the manifests
        kubectl apply -f k8s/deployment.yaml
        kubectl apply -f k8s/service.yaml

        # Get the LoadBalancer URL for your API
        kubectl get service churn-api-service
        ```

7.  **Access Services**:

      * **MLflow UI**: `http://<MLflow-LoadBalancer-IP>`
      * **Prediction API**: `http://<API-LoadBalancer-IP>`

-----

## 🧹 Clean Up Resources

To avoid incurring future charges, it is crucial to delete the AWS resources you created.

```bash
# Set your UNIQUE_ID and AWS_REGION if not already set
export UNIQUE_ID="[YOUR-UNIQUE-ID]"
export AWS_REGION="ap-south-1"

# Delete the EKS cluster
eksctl delete cluster --name churn-mlops --region $AWS_REGION

# Delete the S3 buckets
aws s3 rb s3://mlops-churn-raw-data-$UNIQUE_ID --force
aws s3 rb s3://mlops-churn-processed-data-$UNIQUE_ID --force
aws s3 rb s3://mlops-churn-model-artifacts-$UNIQUE_ID --force

# Delete the SageMaker Pipeline and Model Group
aws sagemaker delete-pipeline --pipeline-name churn-pipeline
# You may need to delete model packages before deleting the group
aws sagemaker delete-model-package-group --model-package-group-name ChurnModelPackageGroup

# Manually delete the following from the AWS Console:
# - IAM Roles: SageMakerChurnRole, LambdaChurnTriggerRole
# - Lambda Function: trigger-churn-pipeline
# - CloudWatch Log Groups associated with the Lambda and SageMaker jobs
# - ECR Repository: churn-prediction-api
# - SageMaker Studio Domain
```

\<div align="center"\>
\<p\>Built with ❤️ for the MLOps community\</p\>
\<p\>⭐ Star this repo if you find it helpful\!\</p\>
\</div\>

```
