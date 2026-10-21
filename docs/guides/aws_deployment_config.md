# AWS deployment configuration

The starter Terraform values are in
[`deploy/aws/terraform.tfvars.example`](../deploy/aws/terraform.tfvars.example).
Copy that file to `deploy/aws/terraform.tfvars` and replace the example image
account ID and client CIDR before planning. The real `.tfvars` file is ignored
by Git; do not put access keys or passwords in it.

## Values already chosen by the project

- Region: `us-west-2` (the existing Terraform and runbook default).
- Project name: `recommender-demo`.
- Model key: `models/model_bundle.zip`.
- Bedrock: optional. The current local `terraform.tfvars` leaves
  `bedrock_model_id` empty, so explanations are disabled.
- SageMaker MLflow: optional. `enable_sagemaker_mlflow` defaults to `false` to
  avoid creating a billable tracking server until it is needed.
- ECS: `deploy_service = false` keeps the API service at zero running tasks
  until a validated model bundle and immutable image are ready.
- Networking: Terraform uses the account's existing default VPC and its
  default subnets. The API security group allows port 8000 only from
  `client_cidrs`.

The initial AWS foundation is applied. Terraform state includes the ECR
repository, private model bucket, ECS cluster/task/service definitions, IAM
roles, security group, and CloudWatch resources. The API is not running yet.

## Values still needed

1. **Model bundle.** Run the full notebook or gated batch pipeline and inspect
   its serving policy and manifest. Upload only the validated bundle to the
   existing private S3 bucket before enabling ECS.
2. **Container image.** Build the API image, push it to the existing ECR
   repository, and replace the `:bootstrap` `container_image` value with its
   immutable image tag or digest.
3. **Optional integrations.** To add SageMaker MLflow tracking and automatic
   SageMaker Model Registry synchronization, set
   `enable_sagemaker_mlflow = true`. To enable explanations, choose an
   available regional Bedrock model and set `bedrock_model_id`. See the
   [SageMaker/Bedrock guide](aws_mlflow_bedrock.md) for IAM, cost, and testing
   steps.

## Deployment sequence

1. The AWS identity was verified in account `405398623518` as user
   `aiengineer`. `deploy/aws/terraform.tfvars` is ignored by Git; never place
   access keys or passwords in it.
2. Run the end-to-end notebook in its documented Spark-capable environment,
   then inspect the bundle manifest and validation gate result.
3. Upload the validated bundle and push the immutable API image to the
   existing S3 bucket and ECR repository.
4. Update `container_image`; set `deploy_service = true` only after both
   artifacts exist. If enabling SageMaker MLflow or Bedrock, configure the
   optional settings and review their plan changes at the same time.
5. Verify the API, CloudWatch logs/dashboard/alarms, and rollback. Record costs
   and stop billable services when finished.

The API currently has no authentication or TLS termination. Keep the CIDR
restricted and do not expose it broadly.
