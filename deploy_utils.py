
import os
import io
import zipfile
from config import DEPLOY_DIR

# ---------------- Deployment helpers ----------------
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def write_deployment_files():
    ensure_dir(DEPLOY_DIR)
    dockerfile = """# Dockerfile for Streamlit app
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""
    # NOTE: You must keep requirements.txt manually updated
    # or generate it with `pip freeze > requirements.txt`
    requirements = """streamlit
pyvis
networkx
pandas
torch
sentence-transformers
spacy
neo4j
PyJWT
"""
    docker_compose = """version: '3.8'
services:
  app:
    build: .
    ports:
      - "8501:8501"
    restart: unless-stopped
"""
    with open(os.path.join(DEPLOY_DIR, "Dockerfile"), "w") as f:
        f.write(dockerfile)
    with open(os.path.join(DEPLOY_DIR, "requirements.txt"), "w") as f:
        f.write(requirements)
    with open(os.path.join(DEPLOY_DIR, "docker-compose.yml"), "w") as f:
        f.write(docker_compose)
    with open(os.path.join(DEPLOY_DIR, "README.md"), "w") as f:
        f.write("Deployment files for Semantic KG Explorer")

    # Create the main requirements.txt for the project root
    with open("requirements.txt", "w") as f:
        f.write(requirements)

    return DEPLOY_DIR
