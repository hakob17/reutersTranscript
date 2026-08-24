# Infrastructure (Terraform)

Production deployment of the speaker-attribution pipeline on AWS:

```
S3 ingest/videos/ ──EventBridge──▶ SQS ──▶ Lambda ingest ──▶ Step Functions
                                                                 │
                     ┌───────────────────────────────────────────┤
                     ▼                                           ▼
        AWS Batch GPU task (g6, Spot,               Lambda attribute
        scale-to-zero): transcribe +               (Claude vision on crops +
        diarize + OpenCV chyron detect  ──work/──▶  speaker attribution)
                                                                 │
                              auto_ok ┌──────────────────────────┤ needs_review
                                      ▼                          ▼
                               Lambda publish ◀── approve ── AwaitReview (paused;
                               (published/ prefix)            SNS → editor →
                                                              Lambda review_callback)
```

Per-video state lives in DynamoDB (`processing → auto_ok|needs_review →
awaiting_review → approved|rejected → published|failed`).

## Deploy

```bash
# 1. Stage the lambda bundles
bash scripts/build_lambdas.sh

# 2. Provision
cd infra
terraform init
terraform apply -var alert_email=you@example.com

# 3. Set the secrets (values never touch Terraform state)
aws secretsmanager put-secret-value \
  --secret-id "$(terraform output -raw anthropic_secret_arn)" \
  --secret-string 'sk-ant-...'
aws secretsmanager put-secret-value \
  --secret-id "$(terraform output -raw hf_token_secret_arn)" \
  --secret-string 'hf_...'

# 4. Build & push the GPU worker image (from the repo root)
ECR=$(terraform -chdir=infra output -raw ecr_repository_url)
aws ecr get-login-password | docker login --username AWS --password-stdin "${ECR%%/*}"
docker build -f services/gpu_task/Dockerfile \
  --build-arg HF_TOKEN=hf_... \
  -t "$ECR:v1" .
docker push "$ECR:v1"

# 5. Process a video
aws s3 cp package.mp4 "s3://$(terraform -chdir=infra output -raw data_bucket)/ingest/videos/"
# optional shotlist: s3://<bucket>/ingest/shotlists/<video_stem>.txt
```

## Review flow

`needs_review` videos pause the workflow and email the review topic. An editor
(or the future review UI) resumes with:

```bash
aws lambda invoke --function-name "$(terraform -chdir=infra output -raw review_callback_function)" \
  --cli-binary-format raw-in-base64-out \
  --payload '{"video_id": "epstein_778738", "approve": true}' /dev/stdout
```

Approved (and `auto_ok`) outputs land under `published/<video_id>/` —
that prefix is what the CMS / player consumes. Rejections and review timeouts
(default 72h, `var.review_timeout_hours`) fail the execution and mark the video
`failed`/`rejected` in DynamoDB.

## Cost shape

Idle cost is ~zero: Batch scales to 0 instances, everything else is
pay-per-request. Per 6-min video ≈ $0.10–0.20 (GPU minutes + two Claude calls);
see the cost table in the demo page for the stage breakdown.

## Notes

- The GPU image bakes model weights at build time; set `HF_HUB_OFFLINE=1` in
  the job definition env once confirmed, removing the runtime HF dependency.
- Spot is on by default (`var.use_spot`); jobs are idempotent and the job
  definition retries 3×, so interruptions only cost minutes.
- Remote state: configure the S3 backend in `versions.tf` before team use.
- Lambda bundles must be re-staged (`build_lambdas.sh`) whenever
  `services/lambdas/` or `speaker_attribution/` change; Terraform picks up the
  new hash automatically.
