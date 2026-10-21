# AWS training and deployment pipeline

This workflow runs the existing training and validation code in a SageMaker
Training Job, stores each run's artifacts in S3, and deploys the selected
serving bundle through the existing ECS/Fargate FastAPI service. SageMaker
Pipelines orchestrates the run; ECS remains the inference host. A failed
two-tower validation gate results in the packaged popularity fallback, as in
the local workflow.

The pipeline expects the **encoded** dataset produced by the existing Spark
preparation commands. Upload a prefix containing `train/`, `validation/`, and
`item_mapping/` to `s3://<model-bucket>/training-data/<version>/`. Raw review
data preparation is still performed in the existing Spark environment; the
SageMaker Training Job performs model training, evaluation, packaging, and
promotion. It uses CPU instances by default.

## One-time AWS setup

Apply the Terraform changes from `deploy/aws` after reviewing the plan. They
add a versioned model bucket, a separate immutable ECR repository for the
training image, and SageMaker pipeline/job roles. The job role can read only
the `training-data/` prefix, write the serving bundle and model artifacts, and
update this project's ECS service. The API service remains at zero tasks until
the training job completes its first run.

Build and push the API image using the existing AWS deployment quick start.
Set `container_image` to its immutable ECR tag, keep `deploy_service = false`,
and apply Terraform. This makes the API task definition ready without starting
an API task. Upload the prepared dataset:

```powershell
$bucket = terraform -chdir=deploy/aws output -raw model_bucket
aws s3 sync .\data\processed\movies_tv_5m\two_tower\v1 "s3://$bucket/training-data/movies-tv-v1/" --region us-west-2
```

Build the SageMaker training image from the repository root and push it to the
new training repository:

```powershell
$env:AWS_PROFILE = "terraform-process"
$trainingRepo = terraform -chdir=deploy/aws output -raw training_ecr_repository_url
$tag = (git rev-parse --short HEAD).Trim()
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $trainingRepo
docker build -f deploy/aws/Dockerfile.sagemaker -t "${trainingRepo}:${tag}" .
docker push "${trainingRepo}:${tag}"
```

Install the SageMaker SDK in the machine that starts the pipeline:

```powershell
pip install "sagemaker>=2.240,<3"
```

The AWS identity starting the pipeline needs SageMaker pipeline create/update,
describe, and start permissions, plus `iam:PassRole` scoped to the pipeline
role. Terraform's pipeline role starts Training Jobs and passes the training
job role. That role has the S3, ECR, CloudWatch Logs, and ECS permissions
needed while the job runs.

## Newcomer walkthrough: train and deploy

This walkthrough is for Windows PowerShell. Start in the repository root.
It assumes Git, Python 3.11, Java 21, AWS CLI v2, Terraform, and Docker Desktop
are installed, and that Docker Desktop is running. The AWS identity must be
allowed to create the Terraform resources in this folder. Do not put AWS keys
in project files.

### 1. Verify AWS access and choose a region

Configure an AWS CLI profile using your organization’s approved method (for
example, AWS IAM Identity Center/SSO), then use its profile name below:

```powershell
$env:AWS_PROFILE = "default"
$region = "us-west-2"
# If this is an IAM Identity Center profile, sign in once per session:
# aws sso login --profile $env:AWS_PROFILE
aws sts get-caller-identity
terraform version
docker version
```

Confirm the returned account is the intended account. Use the same region for
Terraform, ECR, S3, and SageMaker. The deployment identity needs permissions
for ECR, S3, IAM, ECS, CloudWatch, and Terraform state/provider access. The
identity that starts SageMaker also needs pipeline create/update/start/describe
and `iam:PassRole` for the pipeline role.

### 2. Prepare the encoded dataset

The cloud job expects prepared Parquet data; it does not download or process
raw reviews. Follow the [repository setup instructions](../README.md#setup)
to download the Movies and TV source data and configure Spark in a Linux/WSL
Python 3.11 + Java 21 environment. In a WSL Bash shell, from the repository
root, run:

```bash
python -m src.data.prepare \
  --input data/raw/Movies_and_TV.jsonl \
  --output data/processed/movies_tv_5m
python -m src.models.prepare_tower_data \
  --data data/processed/movies_tv_5m \
  --output data/processed/movies_tv_5m/two_tower/v1
ls data/processed/movies_tv_5m/two_tower/v1
```

Confirm that the output contains `train`, `validation`, and `item_mapping`
directories. These preparation commands need Spark and sufficient local
memory; they can take a while. Continue with the PowerShell commands below,
using the same repository folder (for example, a repo under `/mnt/c/...` is
also visible at its corresponding `C:\...` path).

### 3. Create the local AWS config and apply the foundation

Copy the example Terraform variables. Open the file and replace
`REPLACE_WITH_YOUR_PUBLIC_IPV4/32` with your current public IPv4 address plus
`/32`. Keep `deploy_service = false`. The example `container_image` value is a
bootstrap placeholder used only while the service is stopped. Ensure the
region is `us-west-2` for this walkthrough.

```powershell
$clientIp = (Invoke-RestMethod "https://checkip.amazonaws.com").Trim()
Write-Host "Use this value in client_cidrs: $clientIp/32"
Copy-Item deploy/aws/terraform.tfvars.example deploy/aws/terraform.tfvars
notepad deploy/aws/terraform.tfvars
terraform -chdir=deploy/aws init
terraform -chdir=deploy/aws fmt -check
terraform -chdir=deploy/aws validate
terraform -chdir=deploy/aws plan
```

Review the plan, account, and region. Continue only if the proposed resources
are expected; then apply:

```powershell
terraform -chdir=deploy/aws apply
```

This creates billable AWS resources. The API desired task count remains zero
until training succeeds.

### 4. Build and push the FastAPI image

Use a unique tag because the ECR repositories reject overwriting an existing
tag:

```powershell
$apiRepo = terraform -chdir=deploy/aws output -raw ecr_repository_url
$tag = "run-$(Get-Date -Format 'yyyyMMddHHmmss')"
aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $apiRepo
docker build -t "${apiRepo}:${tag}" .
if ($LASTEXITCODE -ne 0) { throw "API image build failed" }
docker push "${apiRepo}:${tag}"
if ($LASTEXITCODE -ne 0) { throw "API image push failed" }
```

Open `deploy/aws/terraform.tfvars`. Set `container_image` to the exact
`$apiRepo` value followed by `:$tag`, and leave `deploy_service = false`.
For example, use the URI printed by Terraform plus the tag printed by:

```powershell
$apiRepo
$tag
```

Then review and apply the task-definition update:

```powershell
terraform -chdir=deploy/aws plan
terraform -chdir=deploy/aws apply
```

### 5. Upload the dataset and push the training image

```powershell
$bucket = terraform -chdir=deploy/aws output -raw model_bucket
$trainingRepo = terraform -chdir=deploy/aws output -raw training_ecr_repository_url
aws s3 sync .\data\processed\movies_tv_5m\two_tower\v1 "s3://$bucket/training-data/movies-tv-v1/" --region $region
if ($LASTEXITCODE -ne 0) { throw "Dataset upload failed" }
aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $trainingRepo
docker build -f deploy/aws/Dockerfile.sagemaker -t "${trainingRepo}:${tag}" .
if ($LASTEXITCODE -ne 0) { throw "Training image build failed" }
docker push "${trainingRepo}:${tag}"
if ($LASTEXITCODE -ne 0) { throw "Training image push failed" }
```

Install the SageMaker SDK in the Python environment used to start the pipeline
(Python 3.11 is recommended):

```powershell
python -m pip install "sagemaker>=2.240,<3"
python -c "import sagemaker; print(sagemaker.__version__)"
```

### 6. Confirm SageMaker quota before launching

The earlier `ml.m5.2xlarge` on-demand request had zero quota, and
`ml.t3.medium` was unavailable for training in `us-west-2`. The Service Quotas
screen values `0 4 0` for `ml.m5.large for spot training job usage` mean the
applied quota is 0, the AWS default quota is 4, and utilization is 0. SageMaker
confirmed the applied limit is 0. Request an applied quota of **1** for
`ml.m5.large for spot training job usage`; this allows this pipeline's single
Spot instance. The pipeline uses Managed Spot by default. Spot capacity can
be interrupted or unavailable. This bounded job does not resume from an
intermediate checkpoint after interruption, so SageMaker may need to restart
it; the six-hour wait limit prevents waiting indefinitely. If you prefer
predictable on-demand capacity, separately request quota 1 for
`ml.m5.large for training job usage` and disable Spot in the estimator.
Check current AWS pricing; Spot can reduce training compute cost substantially,
but savings and availability vary.

### 7. Start the pipeline

The command uses one managed Spot `ml.m5.large`, a 30 GB training volume, and
a maximum six-hour window to obtain capacity and finish. The pipeline trains
and evaluates the model, copies its serving bundle to S3, and then asks ECS
to run one API task:

```powershell
$pipelineRole = terraform -chdir=deploy/aws output -raw sagemaker_pipeline_role_arn
$trainingRole = terraform -chdir=deploy/aws output -raw sagemaker_training_role_arn
$cluster = terraform -chdir=deploy/aws output -raw ecs_cluster_name
$service = terraform -chdir=deploy/aws output -raw ecs_service_name

python deploy/aws/pipeline.py `
  --region $region `
  --pipeline-role-arn $pipelineRole `
  --training-role-arn $trainingRole `
  --training-image "${trainingRepo}:${tag}" `
  --data-s3-uri "s3://$bucket/training-data/movies-tv-v1/" `
  --model-s3-uri "s3://$bucket/models/model_bundle.zip" `
  --ecs-cluster $cluster `
  --ecs-service $service `
  --instance-type ml.m5.large `
  --instance-volume-gb 30 `
  --wait
```

### 8. Verify the task and API

The task has a public IP, while its security group accepts port 8000 only from
the CIDR configured in `client_cidrs`. Find its address and call the health,
model, and recommendation endpoints:

```powershell
aws ecs describe-services --cluster $cluster --services $service --region $region
$taskArn = aws ecs list-tasks --cluster $cluster --service-name $service --region $region --query 'taskArns[0]' --output text
$eni = aws ecs describe-tasks --cluster $cluster --tasks $taskArn --region $region --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value | [0]' --output text
$apiIp = aws ec2 describe-network-interfaces --network-interface-ids $eni --region $region --query 'NetworkInterfaces[0].Association.PublicIp' --output text
Invoke-RestMethod "http://${apiIp}:8000/ready"
Invoke-RestMethod "http://${apiIp}:8000/model-info"
Invoke-RestMethod "http://${apiIp}:8000/recommendations/0?limit=10"
```

### 9. Troubleshoot a failed run

The pipeline command prints an execution ARN. Check pipeline executions,
Training Jobs, ECS events, and API logs with:

```powershell
aws sagemaker list-pipeline-executions --pipeline-name recommender-cloud-training --region $region
aws sagemaker list-training-jobs --region $region --max-results 10
aws ecs describe-services --cluster $cluster --services $service --region $region
aws logs describe-log-streams --log-group-name /ecs/recommender-demo --region $region
```

If execution fails before a Training Job appears, check quota, regional
instance availability, and pipeline-role/`iam:PassRole` permissions. If a job
starts and fails, read its failure reason and its SageMaker CloudWatch logs.
If training succeeds but the API does not become healthy, inspect ECS service
events and `/ecs/recommender-demo` logs.

### 10. Stop the service and clean up

To stop the API task, set `deploy_service = false` in
`deploy/aws/terraform.tfvars`, then review and apply:

```powershell
terraform -chdir=deploy/aws plan
terraform -chdir=deploy/aws apply
```

This keeps the S3 bucket, images, logs, and other resources, which may still
incur storage or service charges. Full removal requires backing up needed
artifacts, emptying the S3 bucket, deleting ECR images, then reviewing and
running `terraform -chdir=deploy/aws destroy`. Never destroy a shared or
production stack.

## Short command reference (for an already configured workspace)

The [newcomer walkthrough above](#newcomer-walkthrough-train-and-deploy) is
the complete first-run path. This shorter reference is for a workspace whose
Terraform foundation, config, images, and dataset are already prepared. Use
the same AWS profile and region as the Terraform setup.

First, verify the identity and review/apply the infrastructure plan:

```powershell
$env:AWS_PROFILE = "terraform-process"
$region = "us-west-2"

aws sts get-caller-identity
terraform -chdir=deploy/aws init
terraform -chdir=deploy/aws plan
# Apply only after reviewing the plan.
terraform -chdir=deploy/aws apply
```

Build and push the FastAPI image, then update Terraform's ignored local
`terraform.tfvars` with its immutable tag. Keep `deploy_service = false` so no
API task starts before the model is ready:

```powershell
$apiRepo = terraform -chdir=deploy/aws output -raw ecr_repository_url
$tag = (git rev-parse --short HEAD).Trim()
aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $apiRepo
docker build -t "${apiRepo}:${tag}" .
docker push "${apiRepo}:${tag}"
```

Set `container_image = "<apiRepo>:<tag>"` and `deploy_service = false` in
`deploy/aws/terraform.tfvars`, substituting the pushed values, then apply the
task-definition update:

```powershell
terraform -chdir=deploy/aws plan
terraform -chdir=deploy/aws apply
```

Upload the encoded training dataset, build and push the SageMaker image, and
install the SDK in the active Python environment:

```powershell
$bucket = terraform -chdir=deploy/aws output -raw model_bucket
$trainingRepo = terraform -chdir=deploy/aws output -raw training_ecr_repository_url
aws s3 sync .\data\processed\movies_tv_5m\two_tower\v1 "s3://$bucket/training-data/movies-tv-v1/" --region $region
aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $trainingRepo
docker build -f deploy/aws/Dockerfile.sagemaker -t "${trainingRepo}:${tag}" .
docker push "${trainingRepo}:${tag}"
python -m pip install "sagemaker>=2.240,<3"
```

Start and wait for the SageMaker Pipeline. Use `ml.m5.large` after confirming
it has quota and is available in the selected region:

```powershell
$pipelineRole = terraform -chdir=deploy/aws output -raw sagemaker_pipeline_role_arn
$trainingRole = terraform -chdir=deploy/aws output -raw sagemaker_training_role_arn
$cluster = terraform -chdir=deploy/aws output -raw ecs_cluster_name
$service = terraform -chdir=deploy/aws output -raw ecs_service_name

python deploy/aws/pipeline.py `
  --region $region `
  --pipeline-role-arn $pipelineRole `
  --training-role-arn $trainingRole `
  --training-image "${trainingRepo}:${tag}" `
  --data-s3-uri "s3://$bucket/training-data/movies-tv-v1/" `
  --model-s3-uri "s3://$bucket/models/model_bundle.zip" `
  --ecs-cluster $cluster `
  --ecs-service $service `
  --instance-type ml.m5.large `
  --instance-volume-gb 30 `
  --wait
```

After a successful run, verify the API task. The following looks up the public
IP assigned to the ECS task; port 8000 is still restricted by `client_cidrs`:

```powershell
aws sagemaker list-pipeline-executions --pipeline-name recommender-cloud-training --region $region
aws sagemaker list-training-jobs --region $region --max-results 5
aws ecs describe-services --cluster $cluster --services $service --region $region
$taskArn = aws ecs list-tasks --cluster $cluster --service-name $service --region us-west-2 --query 'taskArns[0]' --output text
$eni = aws ecs describe-tasks --cluster $cluster --tasks $taskArn --region us-west-2 --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value | [0]' --output text
$apiIp = aws ec2 describe-network-interfaces --network-interface-ids $eni --region us-west-2 --query 'NetworkInterfaces[0].Association.PublicIp' --output text
Invoke-RestMethod "http://${apiIp}:8000/ready"
Invoke-RestMethod "http://${apiIp}:8000/model-info"
```

To stop the running API after the demo, set `deploy_service = false` in
`terraform.tfvars`, then review and apply the plan. Training instances stop
when their jobs finish; the S3 artifacts and ECR images remain:

```powershell
terraform -chdir=deploy/aws plan
terraform -chdir=deploy/aws apply
```

## Start training and deployment

Run from the repository root after the Terraform apply and image upload:

```powershell
$bucket = terraform -chdir=deploy/aws output -raw model_bucket
$pipelineRole = terraform -chdir=deploy/aws output -raw sagemaker_pipeline_role_arn
$trainingRole = terraform -chdir=deploy/aws output -raw sagemaker_training_role_arn
$cluster = terraform -chdir=deploy/aws output -raw ecs_cluster_name
$service = terraform -chdir=deploy/aws output -raw ecs_service_name
$trainingRepo = terraform -chdir=deploy/aws output -raw training_ecr_repository_url
$tag = (git rev-parse --short HEAD).Trim()

python deploy/aws/pipeline.py `
  --region us-west-2 `
  --pipeline-role-arn $pipelineRole `
  --training-role-arn $trainingRole `
  --training-image "${trainingRepo}:${tag}" `
  --data-s3-uri "s3://$bucket/training-data/movies-tv-v1/" `
  --model-s3-uri "s3://$bucket/models/model_bundle.zip" `
  --ecs-cluster $cluster `
  --ecs-service $service `
  --wait
```

The SageMaker pipeline creates a managed Training Job. It runs the same
train/evaluate/export batch workflow used locally, stores the run directory
and MLflow database in SageMaker's S3 model artifact, uploads the selected
bundle to the API's model key, and asks ECS to start one task with the new
bundle. ECS replaces the FastAPI task using its deployment circuit breaker.
The quality gate and selected serving policy are recorded in
`batch_manifest.json` and `validation_gate.json` in the run artifacts. The
job's model archive is also retained under the `training-job-output/` S3
prefix.

Inspect the pipeline execution and Training Job in SageMaker, its model
archive in S3, and the task logs in CloudWatch Logs. Then check the
restricted API endpoint's `/ready`, `/model-info`, and recommendation route.
The API's existing CIDR restriction still applies; it has no authentication
or TLS termination.

## Operational notes

- Training runs on one managed Spot `ml.m5.large` by default, uses a 30 GB
  volume, has a four-hour runtime limit, and can wait up to six hours for
  capacity and completion. Change these only after sizing the dataset and
  checking current SageMaker pricing.
- The SageMaker job uploads the model bundle only after the batch command has
  completed and produced its manifest. S3 versioning keeps prior bundle
  versions available for rollback.
- The job requests one ECS task after promotion. To keep the API stopped after
  training, change `sagemaker_train.py` to skip the ECS update or provide a
  separate training-only pipeline before starting it.
- MLflow uses a run-local SQLite tracking store inside the Training Job;
  that database and its artifacts are retained in the S3 model archive.
  The existing SageMaker-managed MLflow server remains a separate optional
  integration and can be connected later.
- The pipeline definition is created or updated by `pipeline.py` and the
  execution starts immediately. Re-run it with the same name to pick up a new
  image, dataset, or parameters.
- Stop the ECS service by setting `deploy_service = false` and applying
  Terraform after the demo. SageMaker Training instances stop when the job
  ends; S3 artifacts and ECR images remain until explicitly cleaned up.

## What was provisioned and attempted

The initial AWS deployment created the versioned S3 bucket, separate ECR
repositories for the API and training images, SageMaker pipeline and job IAM
roles, and the ECS/Fargate resources. The API image and the training image
were built and pushed to ECR, and the prepared dataset was uploaded under
`training-data/movies-tv-v1/`. The ECS task definition was pointed at the API
image, while the ECS service was left at zero tasks until a model is trained.

The SageMaker Pipeline was created and two executions were attempted:

1. The initial `ml.m5.2xlarge` training instance was rejected because this
   account has zero SageMaker training quota for that instance type in
   `us-west-2`.
2. A retry with `ml.t3.medium` was rejected because that instance type is not
   available for SageMaker Training Jobs in `us-west-2`.

Both executions failed before SageMaker started a Training Job. No model has
been trained, copied to the serving model key, or deployed to ECS; the ECS
service remains stopped. The AWS resources, pushed images, uploaded dataset,
and pipeline are ready for another run once an available training instance
with sufficient account quota is selected. The account's current applied Spot
quota is 0 (AWS default 4), so request applied quota 1 for
`ml.m5.large for spot training job usage`, then rerun the pipeline command
above. If Spot capacity is unavailable or the low-memory instance cannot
complete training, request on-demand quota 1 for
`ml.m5.large for training job usage` or one `ml.m5.xlarge` instance.

The source replacement is `src/pipeline/sagemaker_train.py`, which replaces
the former `src/pipeline/sagemaker_job.py`. The `project_tests.ipynb`
notebook was intentionally left unstaged in Git.

For SageMaker's custom-container contract, see AWS's guides to [training
images](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-dockerfile.html)
and [training information and output directories](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html).
