# AWS deployment quick start

This is the short checkpoint for the guided deployment. See
[`aws_deployment_runbook.md`](aws_deployment_runbook.md) for the longer design,
validation, monitoring, and cleanup notes.

## Current checkpoint — October 15, 2026

- AWS CLI identity was verified in account `405398623518` as IAM user `aiengineer`.
- Terraform is configured for `us-west-2`, project `recommender-demo`, and a
  narrow client CIDR in the ignored local `deploy/aws/terraform.tfvars` file.
- The latest Terraform plan reports **16 to add, 0 to change, 0 to destroy**.
- The planned resources are: one security group; one ECR repository; one
  encrypted, private S3 bucket and its two protection/encryption settings; one
  CloudWatch log group, dashboard, and two alarms; one ECS cluster, task
  definition, and service; two IAM roles; and two IAM inline policies.
- The ECS service's desired task count is **0** (`deploy_service = false`).
  Applying this plan creates the deployment foundation, but does not start the
  API or a Fargate task. Some AWS resources can still have small ongoing
  charges; see the AWS pricing links in the full runbook.
- An earlier `terraform apply` was blocked on missing
  `ecs:CreateCluster` and `ecs:RegisterTaskDefinition` permissions. A later
  apply completed successfully. Its final output reported 3 added, 2 replaced
  CloudWatch alarms destroyed, and no changes to other resources.
- Terraform state now lists the full 16-resource stack, including the ECS
  cluster, task definition, and service. The local configuration still sets
  `deploy_service = false`, so the desired task count is 0 and no Fargate task
  should be running. The image value is still the `:bootstrap` placeholder.

## Permission needed to continue

Ask the AWS administrator for the `AlexAccess` group to grant the deployment
identity the ECS permissions needed to create/manage this project's cluster,
task definition, and service. The first denied actions were
`ecs:CreateCluster` and `ecs:RegisterTaskDefinition`; ECS will also need to
create and inspect the service. Terraform must pass only these two task roles
to ECS:

```text
arn:aws:iam::405398623518:role/recommender-demo-execution
arn:aws:iam::405398623518:role/recommender-demo-task
```

Scope `iam:PassRole` to those role ARNs and the `ecs-tasks.amazonaws.com`
service principal. AWS documents this service-principal condition for ECS task
roles in its [ECS IAM service authorization reference](https://docs.aws.amazon.com/service-authorization/latest/reference/list_ecs.html).
Do not solve this by granting `AdministratorAccess`. Terraform may report any
additional missing action after the first blockers are addressed; review those
errors and keep the policy scoped.

This permission issue is resolved for the latest successful apply. If Terraform
reports a future access-denied error, pause and request only the additional
scoped permission. To verify the current state, from `deploy/aws` run:

```powershell
$env:AWS_PROFILE = "terraform-process"
terraform plan
```

The plan should be a no-op while the current configuration is unchanged. Review
any proposed changes before applying.

## Step 1: verify the deployment foundation

From the repository root in PowerShell:

```powershell
$env:AWS_PROFILE = "terraform-process"
Set-Location .\deploy\aws
terraform plan
```

The profile uses the AWS CLI credential process; it keeps access keys out of
this repository. The infrastructure foundation is already applied. The plan
should be a no-op; if it differs, stop and review the new plan before applying.

## Step 2: prepare and upload the model bundle

Run the full walkthrough in [`end_to_end_recommender.ipynb`](end_to_end_recommender.ipynb)
in the documented Spark-capable environment. Verify that its selected serving
policy passes the validation gate and that it exports a complete
`model_bundle.zip`. Do **not** use the weak, two-epoch Airflow smoke artifact as
the deployment model. Upload only the validated bundle to the S3 bucket and
key configured in `deploy/aws/terraform.tfvars`:

```powershell
$bucket = terraform output -raw model_bucket
aws s3 cp <validated-model-bundle.zip> "s3://$bucket/models/model_bundle.zip" --region us-west-2
```

Replace `<validated-model-bundle.zip>` with the actual path to the notebook's
exported bundle. Do not paste credentials into commands or this file.

## Step 3: push the API image

From the repository root, set the profile if opening a new PowerShell window,
then run:

```powershell
$env:AWS_PROFILE = "terraform-process"
$repo = terraform -chdir=deploy/aws output -raw ecr_repository_url
$tag = (git rev-parse --short HEAD).Trim()
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $repo
docker build -t "${repo}:${tag}" .
docker push "${repo}:${tag}"
```

The ECR repository uses immutable image tags. If that Git-derived tag already
exists, choose a new unique tag rather than overwriting it.

## Step 4: start the API

After both the validated model bundle and image are present, edit the ignored
`deploy/aws/terraform.tfvars` file:

```hcl
container_image = "<ECR repository URL>:<pushed immutable tag>"
deploy_service  = true
```

Then review and apply the second plan:

```powershell
Set-Location .\deploy\aws
terraform plan
terraform apply
```

This starts one Fargate task and incurs runtime charges while it is running.
Check ECS service events and CloudWatch logs if it does not become healthy. Once
healthy, verify `/ready`, `/model-info`, and a recommendation request using the
API security group's restricted client CIDR. The API is plain HTTP and has no
application authentication; use only non-sensitive demo data.

## Useful outputs and cleanup

For the complete SageMaker MLflow tracking/model-registry and Bedrock
explanation workflow, see the [AWS MLflow and Bedrock guide](aws_mlflow_bedrock.md).

From `deploy/aws`, these Terraform outputs identify the ECR repository, model
bucket, ECS cluster/service, and CloudWatch dashboard:

```powershell
terraform output
```

After testing, stop the task by setting `deploy_service = false` and applying,
or clean up the whole experiment using the cleanup procedure in the full
runbook. Review a fresh Terraform plan before either action.
