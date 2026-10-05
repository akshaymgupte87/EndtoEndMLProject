# AWS deployment experiment: tracking, deployment, and monitoring

This runbook explains the recommender's optional SageMaker MLflow experiment
and its minimal AWS serving demo using ECS/Fargate. The stack is intentionally
small: ECS, ECR, S3, IAM, and CloudWatch. It uses the account's existing default
VPC; Terraform does not create a VPC, load balancer, DNS records, or TLS
certificate. The task is reachable directly at its public IP over HTTP on
port 8000. Use only non-sensitive demo data. AWS resources are billable; review
the Terraform plan and a separate estimate before applying. No AWS resources
have been created as part of writing this guide.

## What the current AWS path does

This is the deployment contract for this repository. You train in WSL, build
the API image from the repository root, and deploy only the API to ECS. The
Spark batch job is **not** deployed by the current Terraform. You upload its
serving bundle to S3 yourself before starting ECS.

```text
Local/WSL batch training and validation → S3 model bundle
ECR image → ECS Fargate task with public IP → recommendation client
                    └── stdout/stderr → CloudWatch Logs
ECS service metrics → CloudWatch dashboard and alarms
```

Terraform creates the ECS cluster/service/task, ECR repository, encrypted and
private S3 model bucket, IAM roles, API security group, CloudWatch log group,
dashboard, and alarms. The task uses 0.5 vCPU and 2 GiB memory. Its task role
can read only the model bundle; its execution role can pull the project's image
and write the task logs. ECS needs VPC networking, so Terraform discovers the
existing default VPC and its subnets and assigns the task a public IP. It does
not create VPC resources. The required `client_cidrs` input limits access to
the public IP/CIDR you provide; use your own IP with `/32`. The configuration
rejects `0.0.0.0/0`.

The API currently requires **no application secrets**. Its AWS CLI SSO
credentials authenticate Terraform and AWS CLI commands on your machine; they
are not copied into the task, image, or repository. The model bundle in S3 is
an artifact, not a secret. There is no Secrets Manager resource or secret
value in this design. API routes have no authentication, and direct public
HTTP access is suitable only for non-sensitive demo data.

Local API tests use a small direct ASGI test helper instead of a separate
HTTP client package:

```powershell
.\venv\Scripts\python.exe -m pytest tests/test_api.py -q
```

Result (2026-10-05): **2 passed**. The tests cover API health/readiness,
known/unknown users, limits, metrics, and Redis-outage fallback; the ASGI test
helper avoids adding an HTTP client dependency.

The current AWS Terraform does **not** deploy Prometheus, Grafana, an
OpenTelemetry Collector, X-Ray tracing, a Redis cache, or a SageMaker MLflow
server. ECS task logs go to CloudWatch Logs, and ECS publishes service
metrics to CloudWatch Metrics. The API's Prometheus-format `/metrics` endpoint
is available in the container, but AWS does not scrape it in the current
configuration. Do not say application metrics or distributed traces are
visible in CloudWatch until we add and test that integration.

This is not a production service: it has one task, public HTTP access, no API
authentication, no dedicated VPC, no autoscaling, no alarm notifications,
local Terraform state, and no AWS-exported application metrics/traces. Its
security group is limited to the client IP CIDR. Logs expire after 14 days;
bucket and ECR deletion are not forced. Estimate Fargate runtime, public IPv4,
CloudWatch log volume, ECR/S3 storage, and data transfer before apply. There is
no NAT Gateway or ALB charge in this design.

## What changes are needed for SageMaker managed MLflow

The training and evaluation CLIs already accept `--mlflow-tracking-uri`; the
shared helper sets that URI before logging. Local runs default to
`sqlite:///mlflow.db`. For a SageMaker managed tracking server, the developer
environment needs the compatible `sagemaker-mlflow` plugin, AWS credentials
with the server's MLflow API permissions, and the tracking server ARN as the
tracking URI. SageMaker's plugin signs MLflow requests with AWS SigV4. Run one
small experiment remotely first; do not remove the local default.

Required project/configuration changes before that experiment:

1. Add `sagemaker-mlflow` as an optional dependency group (for example,
   `uv sync --extra sagemaker-mlflow`) and pin a plugin/client combination
   compatible with the tracking server's MLflow version. The current project
   allows MLflow 3.x; select an actually supported server/client pair when
   creating the server and verify it with a small run.
2. Keep the default local URI and pass a remote URI explicitly. Never commit
   an account ID, tracking-server ARN, access key, or secret. Use an AWS CLI
   profile or IAM role for credentials. SageMaker tracking-server ARNs are
   not passwords, but keep them in local environment/configuration rather
   than hard-coding them into source.
3. Ensure the tracking server role can write artifacts to the chosen S3
   bucket, and the caller's IAM identity can call the required
   `sagemaker-mlflow` APIs. Limit both policies to this experiment's server
   and bucket where possible.
4. Record remote artifacts in S3 and test model upload/download. AWS
   documents a 200 MB MLflow model-download size limit for SageMaker MLflow.
   Measure actual artifact/model sizes; the 16.9 MB batch smoke bundle is
   under that limit, but that does not prove future bundles or all artifact
   operations will fit.
5. `train_two_tower.py` and `evaluate_two_tower.py` already have
   `--mlflow-tracking-uri`; no training-code rewrite is required. Add the
   optional plugin dependency and lock file change, then pass the ARN to both
   commands. `register_model.py` separately has `--tracking-uri`; use it for
   the remote registry only after the small-run test. This code creates a
   registered version and assigns the `candidate` alias. Confirm the version
   and alias in the managed UI before treating the remote registry as
   authoritative. No model-serving code changes are needed because ECS loads
   the S3 bundle, not a model directly from SageMaker MLflow.

SageMaker managed MLflow is for this PyTorch recommender's training,
validation, and model-artifact tracking. Bedrock is not a substitute. Bedrock
would be relevant only if the product later invokes a Bedrock foundation model;
then Bedrock evaluation, invocation logs, and Bedrock CloudWatch metrics can
track that separate feature.

### Create and test one managed tracking server

First check the AWS account, region, permissions, pricing, and current service
availability. SageMaker MLflow tracking servers require an artifact S3 bucket
in your account and a service role that can access it. The server itself is a
billable resource. Use a dedicated development bucket and a small server for
this one experiment.

Create a dedicated AWS CLI profile with IAM Identity Center. Replace the SSO
start URL and region with the values supplied by your account administrator.
The profile keeps credentials out of the repository. Use the same AWS region
for SageMaker MLflow and its artifact bucket:

```bash
aws configure sso --profile recommender-dev
aws sso login --profile recommender-dev
export AWS_PROFILE=recommender-dev
aws sts get-caller-identity
MLFLOW_BUCKET='replace-with-a-globally-unique-bucket-name'
AWS_REGION='us-west-2'
aws s3 mb "s3://${MLFLOW_BUCKET}" --region "$AWS_REGION"
```

For an S3 bucket outside `us-east-1`, `aws s3 mb` applies the region
configuration. If the name already exists globally, choose another unique
bucket name and retry. Confirm the bucket region with
`aws s3api get-bucket-location --bucket "$MLFLOW_BUCKET"`.

Create or select two IAM roles/policies before opening Studio. The
**tracking-server service role** is assumed by SageMaker; it needs S3 read,
write, and list access on the artifact bucket, plus the specific SageMaker
Model Registry actions if registering models. The **caller identity** (your
IAM Identity Center role/profile) needs `sagemaker-mlflow` run/log/artifact
actions scoped to this tracking server. If you will create/delete the server,
it also needs the relevant `sagemaker:CreateMlflowTrackingServer`,
`DescribeMlflowTrackingServer`, `StopMlflowTrackingServer`, and
`DeleteMlflowTrackingServer` control-plane actions. Use the AWS IAM policy
guide to build narrow policies; do not attach `AdministratorAccess` as a
workaround for an access error.

In the current SageMaker Studio console, use this route:

1. Open **Amazon SageMaker AI → Studio**, choose the new Studio experience,
   then **Applications → MLflow**.
2. In **MLflow Tracking Servers**, choose **Create** and provide a unique
   server name and the artifact bucket URI created above. The bucket and
   tracking server must be in the same AWS Region.
3. Choose **Configure** to select the tracking-server service role, small
   server size, supported MLflow server version, and whether automatic model
   registration is enabled. For this first experiment, avoid automatic
   production deployment; registration is optional.
4. Create the server, wait until it is active, then record its ARN and launch
   the MLflow UI. If creation stalls, check IAM role trust/access and server
   events before retrying.

AWS currently documents that server creation can take up to about 25 minutes.
Use the official console/CLI guides below for any UI changes and the exact
permissions required by the selected setup.

Install the compatible plugin in the WSL project environment, then run the
existing bounded training and evaluation commands with the ARN as their
tracking URI:

```bash
uv add --optional sagemaker-mlflow sagemaker-mlflow
uv sync --extra sagemaker-mlflow
export RECOMMENDER_MLFLOW_TRACKING_URI='PASTE_THE_TRACKING_SERVER_ARN_HERE'
export AWS_PROFILE=recommender-dev

python -m src.models.train_two_tower \
  --data data/processed/movies_tv_5m/two_tower/v1 \
  --output artifacts/sagemaker_mlflow_smoke \
  --max-users 500 --max-train-pairs 30000 --epochs 2 --seed 42 \
  --mlflow-tracking-uri "$RECOMMENDER_MLFLOW_TRACKING_URI"

python -m src.models.evaluate_two_tower \
  --data data/processed/movies_tv_5m/two_tower/v1 \
  --checkpoint artifacts/sagemaker_mlflow_smoke/best_model.pt \
  --output artifacts/sagemaker_mlflow_smoke/validation_metrics.json \
  --k 10 20 \
  --mlflow-tracking-uri "$RECOMMENDER_MLFLOW_TRACKING_URI"
```

Replace the ARN value with the one shown in Studio. Then open the managed
MLflow UI, find experiment `two-tower-recommender`, and check that the new
training and evaluation runs contain parameters, epoch metrics, final ranking
metrics, and artifacts. Download `history.json`, the checkpoint, and the
validation report; compare file hashes and numeric values with the local
files. The managed run is a pass only if both runs exist and their inputs,
metrics, and files agree with the local execution. After selecting a tracking
server version, pin compatible MLflow/plugin versions in the optional
dependency group and lock them so later runs do not silently change clients.

If registration is enabled, test the gate on a validation report and confirm
the registered version and `candidate` alias in the managed UI. Keep the local
SQLite tracking store as the default/fallback.

When the remote experiment is finished, set the exact server name and region
created for it, check the server status, stop it, and delete it:

```bash
TRACKING_SERVER_NAME='your-server-name'
aws sagemaker describe-mlflow-tracking-server \
  --tracking-server-name "$TRACKING_SERVER_NAME" --region "$AWS_REGION"
aws sagemaker stop-mlflow-tracking-server \
  --tracking-server-name "$TRACKING_SERVER_NAME" --region "$AWS_REGION"
aws sagemaker delete-mlflow-tracking-server \
  --tracking-server-name "$TRACKING_SERVER_NAME" --region "$AWS_REGION"
```

Wait for deletion before removing the tracking-server service role. If the
artifact bucket was created only for this disposable experiment and contains
nothing to retain, remove it explicitly:

```bash
aws s3 rm "s3://${MLFLOW_BUCKET}" --recursive
aws s3 rb "s3://${MLFLOW_BUCKET}" --region "$AWS_REGION"
```

## Prepare and validate the model bundle locally

Use the existing WSL Spark/Python setup and the already prepared mapped data.
Run the bounded batch job to produce the checkpoint, validation report,
serving files, checksums, and `model_bundle.zip`:

```bash
python -m src.pipeline.batch_job \
  --data data/processed/movies_tv_5m/two_tower/v1 \
  --output artifacts/aws_candidate/run1 \
  --max-users 2000 --max-train-pairs 100000 --epochs 10 --seed 42
```

Inspect `artifacts/aws_candidate/run1/batch_manifest.json`,
`artifacts/aws_candidate/run1/validation_metrics.json`, and
`artifacts/aws_candidate/run1/serving/manifest.json`. The batch command writes
`model_bundle.zip` to the output directory's parent, so its path is
`artifacts/aws_candidate/model_bundle.zip`. Apply the existing
validation gate before treating the checkpoint as a candidate:

```bash
python -m src.models.register_model \
  --checkpoint artifacts/aws_candidate/run1/best_model.pt \
  --validation-report artifacts/aws_candidate/run1/validation_metrics.json \
  --registered-name amazon-reviews-two-tower
```

For now the batch job trains and evaluates with MLflow logs. The register
command has its own `--tracking-uri` option; pass the SageMaker ARN explicitly
only after confirming the registry behavior in the managed server. The serving
bundle exported by the batch job is what the current ECS app reads from S3;
ECS does not load the model from MLflow at request time.

## Build and test the Docker image

Install Docker Desktop and ensure its Linux-container engine is running. From
the repository root, point Compose at the serving directory produced above:

```powershell
$env:MODEL_DIR = "./artifacts/aws_candidate/run1/serving"
docker compose config --quiet
docker compose up --build -d
docker compose ps
Invoke-RestMethod http://localhost:8000/ready
Invoke-RestMethod http://localhost:8000/model-info
```

Request a known integer `user_idx` found in `serving/user_ids.json`, inspect
Prometheus at `http://localhost:9090`, and confirm a request trace in
`docker compose logs otel-collector`. Exercise Redis fallback using the
documented cache outage step in [the deployment walkthrough](deployment_walkthrough.md#run-the-api-and-local-observability).
Then stop the stack:

```powershell
docker compose down
```

Build the image explicitly and give it a unique immutable tag. The repository
Dockerfile currently installs unpinned packages at build time; before relying
on repeatable AWS builds, change it to install the locked project runtime
dependencies (or a generated, pinned requirements set), then rebuild and
retest. Save the exact image tag or digest with the experiment record.

```powershell
docker build -t recommender-api:dev .
```

## Deploy the API to AWS with Terraform

### Prerequisites and preflight

- AWS account and CLI access to create ECS, ECR, S3, IAM, CloudWatch, and a security group in the existing default VPC.
- Existing default VPC with at least one subnet that routes to an internet gateway. Terraform reads this VPC and its subnets; it creates no VPC resources.
- Docker Desktop with its Linux engine running, Terraform 1.6+, and the tested model bundle.
- Use only non-sensitive demo data. This deployment is direct public HTTP with no application authentication. Restrict the security group to your current public IPv4 address using `/32`.

Authenticate with AWS IAM Identity Center and check the selected account and network:

```powershell
aws configure sso --profile recommender-dev
aws sso login --profile recommender-dev
$env:AWS_PROFILE = "recommender-dev"
$region = "us-west-2"
$projectName = "recommender-demo"
$clientCidr = "203.0.113.10/32" # replace with your own public IPv4 address /32
$env:TF_VAR_client_cidrs = "[`"$clientCidr`"]"
aws sts get-caller-identity
aws ec2 describe-vpcs --region $region --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text
aws ec2 describe-subnets --region $region --filters Name=default-for-az,Values=true --query 'Subnets[*].[SubnetId,AvailabilityZone,MapPublicIpOnLaunch]' --output table
terraform -version
docker info
```

Replace the example CIDR with your current public IP `/32`; if it changes,
update the variable and reapply the security group. The selected default VPC
must have public subnets and an internet-gateway route. If your account has no
default VPC, this intentionally small Terraform configuration cannot run until
you restore/create a default VPC or choose to add custom VPC resources.

### Create ECR and the model bucket

From the repository root, initialize and validate Terraform, then review the
bootstrap plan. This creates only the ECR repository and private model bucket:

```powershell
Set-Location deploy/aws
terraform init
terraform fmt -check
terraform validate
terraform plan -target=aws_ecr_repository.api -target=aws_s3_bucket.models `
  -var="aws_region=$region" -var="project_name=$projectName" `
  -var="container_image=placeholder"
```

After reviewing the plan, create those two resources:

```powershell
terraform apply -target=aws_ecr_repository.api -target=aws_s3_bucket.models `
  -var="aws_region=$region" -var="project_name=$projectName" `
  -var="container_image=placeholder"
$ecr = terraform output -raw ecr_repository_url
$bucket = terraform output -raw model_bucket
```

The model bucket blocks public access and uses server-side encryption. The ECS
task role can read only `models/model_bundle.zip`. Upload the validated batch
bundle, build the image, and push it with a unique immutable tag:

```powershell
Set-Location ..\..
$accountId = aws sts get-caller-identity --query Account --output text
$tag = "recommender-$(Get-Date -Format yyyyMMdd-HHmmss)"
aws s3 cp .\artifacts\aws_candidate\model_bundle.zip "s3://$bucket/models/model_bundle.zip"
$bundleSha256 = (Get-FileHash .\artifacts\aws_candidate\model_bundle.zip -Algorithm SHA256).Hash
Write-Output "Local model bundle SHA256: $bundleSha256"
aws ecr get-login-password --region $region | docker login --username AWS --password-stdin "$accountId.dkr.ecr.$region.amazonaws.com"
docker build --platform linux/amd64 -t "${ecr}:$tag" .
docker push "${ecr}:$tag"
aws ecr describe-images --repository-name $projectName --image-ids imageTag=$tag --query 'imageDetails[0].imageDigest' --output text
```

Record the image digest and local bundle SHA-256 with the MLflow run IDs.

### Estimate cost, review the full plan, then deploy

Terraform plan lists changes but does not estimate dollars. Use the
[AWS Pricing Calculator](https://calculator.aws/) for one Linux/x86 Fargate
task (0.5 vCPU, 2 GiB) at the number of hours you expect it to run, one public
IPv4 address while the task runs, CloudWatch logs/alarms/dashboard, ECR/S3
storage, and expected outbound data transfer. There is no ALB, NAT Gateway,
Route 53 hosted zone, ACM certificate, or Secrets Manager charge in this
architecture. The AWS public IPv4 price is hourly; stop/destroy the service
when finished. Save the calculator estimate with the Terraform plan. See
[Fargate pricing](https://aws.amazon.com/fargate/pricing/),
[VPC/public IPv4 pricing](https://aws.amazon.com/vpc/pricing/), and
[CloudWatch pricing](https://aws.amazon.com/cloudwatch/pricing/).

Then plan the whole stack. Confirm the plan contains the task/service,
security group, IAM roles, log group, dashboard, alarms, ECR, and S3; check
that Terraform is only reading the existing default VPC/subnets. Do not apply
until you have reviewed the plan and estimate:

```powershell
Set-Location deploy/aws
terraform plan -var="aws_region=$region" -var="project_name=$projectName" `
  -var="container_image=${ecr}:$tag"
```

After review, apply the same inputs:

```powershell
terraform apply -var="aws_region=$region" -var="project_name=$projectName" `
  -var="container_image=${ecr}:$tag"
```

Find the running task's public IP and test the API. The task may receive a new
IP after a redeploy, so repeat discovery after replacing the task:

```powershell
$taskArn = aws ecs list-tasks --region $region --cluster $projectName `
  --service-name $projectName --query 'taskArns[0]' --output text
$eniId = aws ecs describe-tasks --region $region --cluster $projectName --tasks $taskArn `
  --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value | [0]' --output text
$apiIp = aws ec2 describe-network-interfaces --region $region --network-interface-ids $eniId `
  --query 'NetworkInterfaces[0].Association.PublicIp' --output text
$api = "http://${apiIp}:8000"
Invoke-RestMethod "$api/ready"
Invoke-RestMethod "$api/model-info"
$knownUser = (Get-Content ..\..\artifacts\aws_candidate\run1\serving\user_ids.json -Raw | ConvertFrom-Json)[0]
Invoke-RestMethod "$api/recommendations/$knownUser`?limit=10"
```

If the service does not start, inspect ECS events and the stopped-task reason,
then CloudWatch logs:

```powershell
aws ecs describe-services --region $region --cluster $projectName --services $projectName `
  --query 'services[0].events[0:10].[createdAt,message]' --output table
aws ecs describe-tasks --region $region --cluster $projectName --tasks $taskArn `
  --query 'tasks[0].[lastStatus,healthStatus,stoppedReason,containers[0].reason]' --output table
```

Common first-start causes are a missing S3 object, wrong object key, insufficient
S3 role permissions, image pull failure, or a subnet without internet routing.
The startup log reports artifact checksum and loading errors.
## CloudWatch: logs, metrics, and API checks

### Container logs

Terraform creates `/ecs/recommender-demo` with 14-day retention. In AWS
Console, open **ECS → Clusters → recommender-demo → Services → recommender-demo
→ Tasks**, select the task, and choose its **Logs** link. Or use **CloudWatch →
Logs → Log groups → `/ecs/recommender-demo`** and open the newest `api/` stream.
Startup should show that the app downloaded `models/model_bundle.zip`, checked
the serving manifest hashes, loaded the checkpoint and mappings, and passed
`/ready`. Look for `AccessDenied`, `NoSuchKey`, checksum mismatches, import
errors, or repeated restarts.

```powershell
aws logs describe-log-streams --region $region --log-group-name /ecs/recommender-demo --order-by LastEventTime --descending --limit 5
aws logs tail /ecs/recommender-demo --region $region --since 30m --follow
```

In **CloudWatch → Logs Insights**, choose `/ecs/recommender-demo` and the test
time range. To inspect recent output:

```sql
fields @timestamp, @logStream, @message
| sort @timestamp desc
| limit 100
```

To find common startup failures:

```sql
fields @timestamp, @logStream, @message
| filter @message like /ERROR|Error|Traceback|AccessDenied|NoSuchKey|checksum/
| sort @timestamp desc
| limit 100
```

The API uses Uvicorn's plain access logs; it does not yet attach structured
request IDs, model versions, latency fields, or trace IDs. Do not log raw user
IDs or recommendation payloads as a debugging shortcut.

### ECS metrics and dashboard

Open the `recommender-demo` CloudWatch dashboard. It charts ECS CPU/memory and
running task count. Under **CloudWatch → Metrics → AWS/ECS**, choose
`ClusterName=recommender-demo` and `ServiceName=recommender-demo` to inspect
`CPUUtilization`, `MemoryUtilization`, and `RunningTaskCount` over the request
test window. The two alarms are high CPU (over 80% for two 5-minute periods)
and no running task (below one for two 1-minute periods). They have no SNS
actions, so they change state in CloudWatch but do not send notifications;
the thresholds are learning examples, not production SLOs.

The API's `/metrics` is Prometheus text available directly from the task, but
this Terraform does not scrape or export it to CloudWatch. Recall@K, NDCG,
unknown-user behavior, and cache behavior remain offline evaluation/local
signals. ECS metrics are infrastructure signals only.

### Request check and experiment record

With `$api` set to `http://<task-public-ip>:8000`, check `/ready`,
`/model-info`, and one known-user recommendation. Use a known mapped integer
from `serving/user_ids.json`; `-1` should return API-level 404. The AWS API has
no application authentication, so only use non-sensitive demo data and keep
the security group on your public `/32` where possible.

Record UTC test time, AWS region/account alias, Terraform state identity,
image tag and digest, model bundle key and SHA-256, MLflow training/evaluation
run IDs, task public IP, request cases/statuses, CloudWatch log stream,
dashboard observations, and any failures/recovery. The public task IP may
change when ECS replaces a task. `/model-info` reports model type/dimensions,
not model version/checksum; use the saved bundle hash to identify this run.
### Traces and API-specific metrics in AWS

The AWS stack has no trace backend. Local OpenTelemetry sends traces only to
the local Collector debug exporter. Prometheus counters and histograms are
available at `/metrics` but are not sent to CloudWatch. Add AWS tracing or
application-metric export only if the project later needs those signals.

## Cleanup and failure recovery

After capturing results, remove the pushed ECR image and uploaded model bundle;
Terraform will not force-delete either. Keep local copies if you need them.

```powershell
aws ecr batch-delete-image --repository-name $projectName --region $region `
  --image-ids imageTag=$tag
aws s3 rm "s3://$bucket/models/model_bundle.zip" --region $region
terraform destroy -var="container_image=${ecr}:$tag"
```

Review the destroy plan before confirming. The model bucket is not versioned,
so deleting the current bundle empties its object contents. Save experiment
artifacts you intend to retain and clean up the SageMaker tracking server and
its temporary artifacts separately. CloudWatch log retention is 14 days.
If `terraform destroy` or apply fails, inspect the specific resource and AWS
event/error first. Avoid manually deleting resources that Terraform still
tracks unless you understand how to reconcile its state.

## Deployment experiment checklist

- [ ] Same bounded model run recorded locally and in managed MLflow, with
      matching parameters, epoch metrics, report, and artifact checksum.
- [ ] Docker image builds locally and serves the selected local bundle.
- [ ] ECR image tag/digest and S3 bundle key/checksum recorded.
- [ ] Terraform plan reviewed before apply; ECS task becomes healthy.
- [ ] Public task IP, restricted client CIDR, `/ready`, `/model-info`, and
      known/unknown-user recommendation cases checked.
- [ ] CloudWatch dashboard, alarm states, logs, and ECS resource metrics
      inspected for the test window.
- [ ] Actual trace/API-metric gaps noted; no claim that CloudWatch receives
      signals that were not configured.
- [ ] Experiment record completed and billable resources cleaned up.

## AWS documentation used

- [SageMaker managed MLflow](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html)
- [Connect MLflow clients to SageMaker tracking servers](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-track-experiments.html)
- [Create a SageMaker MLflow tracking server](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-create-tracking-server.html)
- [Create a tracking server in SageMaker Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-create-tracking-server-studio.html)
- [SageMaker MLflow IAM permissions](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-create-tracking-server-iam.html)
- [SageMaker MLflow cleanup](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-cleanup.html)
- [ECS CloudWatch logging](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/using_awslogs.html)
- [CloudWatch metrics for ECS](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/cloudwatch-metrics.html)
- [CloudWatch Logs Insights query syntax](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/CWL_QuerySyntax.html)
- [ECS task networking](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task-networking.html)
- [AWS Pricing Calculator](https://calculator.aws/)
- [AWS Fargate pricing](https://aws.amazon.com/fargate/pricing/)
- [CloudWatch pricing](https://aws.amazon.com/cloudwatch/pricing/)
