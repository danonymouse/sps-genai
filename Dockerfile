# Use an official Python runtime as a parent image
FROM python:3.12-slim-bookworm

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest uv installer, run it, then remove it
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Set the working directory
WORKDIR /code

# Copy the dependency files first (better layer caching)
COPY pyproject.toml uv.lock /code/

# Install dependencies only (including spaCy and the en_core_web_lg model).
# --no-install-project: skip building the sps-genai package itself, since
# its source (src/, README.md) isn't copied into the image.
RUN uv sync --frozen --no-install-project

# Copy the application code
COPY ./app /code/app

# Run the app. --no-sync stops uv from trying to install the project at startup.
CMD ["uv", "run", "--no-sync", "fastapi", "run", "app/main.py", "--port", "80"]