# Optional SageMaker MLflow and Bedrock integrations

This guide adds both AWS integrations to the existing ECS recommender without
changing the ranking model or creating a SageMaker inference endpoint.

## What runs where

- The notebook continues to prepare data with local/WSL Spark and train the
  bounded PyTorch model. With an ARN configured, it sends training history,
  validation metrics, and artifacts to a SageMaker-managed MLflow tracking
  server. A model is registered only when the validation gate passes.
- SageMaker MLflow automatically synchronizes registered MLflow models to the
  SageMaker Model Registry. The registry version is for review and lineage;
  ECS still serves the checksummed `model_bundle.zip` from S3.
- Bedrock generates an optional explanation for the items already selected by
  the recommender. It does not rank or replace recommendations.
- The SageMaker Training Job pipeline in
  [the cloud pipeline guide](aws_cloud_ml_pipeline.md) runs batch training and
  refreshes the ECS API when launched. No SageMaker real-time endpoint or
  Bedrock resource is created by default. The tracking server is optional and
  billable while running. Bedrock is invoked on demand by the API route.

## Enable the optional infrastructure

The Terraform defaults leave SageMaker MLflow off and Bedrock unconfigured. In
the ignored `deploy/aws/terraform.tfvars`, set:

```hcl
enable_sagemaker_mlflow = true
bedrock_model_id        = "amazon.nova-micro-v1:0"
deploy_service          = false
```

`amazon.nova-micro-v1:0` is an example regional Bedrock model ID available in
`us-west-2`; confirm model availability and account permissions in the Bedrock
console before using it. The Terraform policy accepts one regional foundation
model ID and scopes `bedrock:InvokeModel` to that model ARN. Cross-region
inference-profile IDs and ARNs are intentionally not accepted by this narrow
configuration.

Then, from `deploy/aws`, authenticate and review the plan:

```powershell
$env:AWS_PROFILE = "terraform-process"
terraform plan
terraform apply
```

With these two options enabled, Terraform creates a small SageMaker MLflow
tracking server, its service role and scoped S3/Model Registry policy, and
updates the ECS task role/task definition with the Bedrock model setting. Since
`deploy_service` remains false, the API task does not start yet. The tracking
server itself is a billable resource; check the current AWS pricing before
leaving it running.

The AWS identity running Terraform needs permission to create/manage this
tracking server and its service role, including `sagemaker:CreateMlflowTrackingServer`,
`DescribeMlflowTrackingServer`, `ListMlflowTrackingServers`, `UpdateMlflowTrackingServer`,
`StartMlflowTrackingServer`, `StopMlflowTrackingServer`, `DeleteMlflowTrackingServer`,
`CreatePresignedMlflowTrackingServerUrl`, and `iam:PassRole` for
`recommender-demo-mlflow-server`. It also needs the corresponding narrowly
scoped IAM role/policy creation permissions. The running tracking server uses
its own role; that role can access only the `mlflow/` S3 prefix plus the SageMaker
Model Registry operations required for registration. Ask the AWS administrator
to grant missing permissions to the deployment identity; do not use
`AdministratorAccess` as a workaround.

## Log experiments and register a passing model

The model-development identity needs `sagemaker-mlflow` actions on this tracking
server ARN. Configure a short-lived AWS CLI profile in the Linux/WSL environment
that runs the notebook; the Windows profile is not automatically shared with
WSL. Do not copy credentials into notebook code. If needed, authenticate in
WSL with `aws login --profile terraform-process --region us-west-2`. In
PowerShell, retrieve the tracking-server ARN with
`terraform -chdir=deploy/aws output -raw sagemaker_mlflow_arn`, then set the
environment variable in the WSL notebook environment. From the project root:

```bash
uv sync --extra sagemaker-mlflow
export AWS_PROFILE=terraform-process
export RECOMMENDER_MLFLOW_TRACKING_URI="<tracking-server-arn-from-terraform-output>"
```

Open `end_to_end_recommender.ipynb` in that environment and run it
through the export cell. It logs one training run and one validation run. If the
validation gate passes, it registers `amazon-reviews-two-tower` with the
`candidate` alias; with the Terraform server setting enabled, SageMaker Model
Registry receives the synchronized model package. If the gate fails, metrics
are still logged, the model is not registered, and the serving bundle selects
popularity.

In the AWS Console, open **Amazon SageMaker AI → MLflow** to inspect runs and
the MLflow Model Registry. Open **Amazon SageMaker AI → Model Registry** to
inspect the synchronized model package. The tracking server uses a small
configuration and an S3 artifact prefix separate from the API bundle. AWS
documents a 200 MB MLflow model-download limit; the model logged by this
project is expected to remain small, but check the actual artifact size.

The existing `train_two_tower.py` and `evaluate_two_tower.py` CLIs also accept
`--mlflow-tracking-uri` for runs outside the notebook. They run on the machine
where invoked. The separately configured cloud pipeline runs batch training
in a SageMaker Training Job; enabling this tracking server by itself does not
move notebook training into AWS.

## Use Bedrock explanations from the API

Bedrock model availability and account permissions vary. In **Amazon Bedrock
→ Model catalog**, confirm the selected regional model supports the Converse
API and that the account can invoke it. The runtime role receives only
`bedrock:InvokeModel` for the configured model. AWS documents that Converse
uses this permission.

After the validated model bundle and immutable API image are uploaded, set
`deploy_service = true`, review the Terraform plan, and apply it. Then, using a
known internal user index from the bundle, send:

```powershell
$api = "http://<task-public-ip>:8000"
$userIdx = 123
Invoke-RestMethod -Method Post -Uri "$api/explain/$userIdx`?limit=5"
```

The route returns the same selected items as `/recommendations/{user_idx}` and
adds cautious generated text. It sends only those recommended item identifiers
to Bedrock; it does not send the user's ID or rating history. Use non-sensitive
demo data. The API is still unauthenticated plain HTTP and must remain limited
to the configured client IP CIDR.

## Stop and clean up

To stop the MLflow server but retain the tracking server and its history, use
the SageMaker CLI with the server name `recommender-demo-mlflow`:

```powershell
aws sagemaker stop-mlflow-tracking-server --tracking-server-name recommender-demo-mlflow --region us-west-2
```

Start it again before logging more runs:

```powershell
aws sagemaker start-mlflow-tracking-server --tracking-server-name recommender-demo-mlflow --region us-west-2
```

Do not set `enable_sagemaker_mlflow = false` unless you intend Terraform to
delete the tracking server. The artifacts remain under the S3 `mlflow/`
prefix, while MLflow tracking metadata is managed by SageMaker. Synchronized
SageMaker Model Registry packages may have their own lifecycle; inspect them
before cleanup. Bedrock incurs inference charges only when the explanation
route invokes the model. Use the AWS Pricing Calculator for current prices.

## References

- [SageMaker managed MLflow overview](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html)
- [Install and connect the SageMaker MLflow plugin](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-track-experiments.html)
- [SageMaker MLflow IAM permissions](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-create-tracking-server-iam.html)
- [Automatically register MLflow models in SageMaker Model Registry](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-track-experiments-model-registration.html)
- [Amazon Nova Micro model ID and Regions](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-micro.html)
- [Bedrock Converse permission](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html)
- [AWS Pricing Calculator](https://calculator.aws/)
