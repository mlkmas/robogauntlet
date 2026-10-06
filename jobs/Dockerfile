# Rollout worker image for Nebius Serverless Jobs.
# Set BASE_IMAGE to an Isaac Lab image that includes Isaac Lab-Arena, e.g. from NGC (nvcr.io).
# Build:  docker build -f jobs/Dockerfile --build-arg BASE_IMAGE=<isaac-lab-image> -t <registry>/robogauntlet-worker .
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY jobs ./jobs
RUN pip install --no-cache-dir .

ENTRYPOINT []
