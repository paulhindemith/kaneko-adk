#!/bin/bash

set -e

# Example usage:
#  ./deploy.sh dtt-gcp test-kaneko test-kaneko-chatgpt-gpts-thelook-ecommerce chatgpt-gpts-thelook-ecommerce

PROJECT_ID="$1"
REPO_NAME="$2"
SERVICE_NAME="$3"
IMAGE_NAME="$4"
API_KEY="$5"

if [ $# -ne 5 ]; then
  echo "Usage: $0 <PROJECT_ID> <REPO_NAME> <SERVICE_NAME> <IMAGE_NAME> <API_KEY>"
  exit 1
fi

AR_PATH="asia-northeast1-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}"

echo "Building Docker image..."
docker build -t ${AR_PATH} . --build-context kaneko-adk=../../../

echo "Pushing Docker image to Artifact Registry..."
docker push ${AR_PATH}

echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image ${AR_PATH} \
  --platform managed \
  --region asia-northeast1 \
  --allow-unauthenticated \
  --project ${PROJECT_ID} \
  --set-env-vars="API_KEY=$API_KEY" \
  --max-instances=1 \
  --min-instances=0 \
  --memory=1Gi \
  --concurrency=3

echo "Deployment successful."
echo "Service Name: ${SERVICE_NAME}"