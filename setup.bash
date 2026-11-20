#! /bin/bash
conda create -n costume-mcp python=3.12 -y
conda run -n costume-mcp pip install torch --index-url https://download.pytorch.org/whl/cpu
conda run -n costume-mcp pip install -r requirements.txt
conda run -n costume-mcp pip install -r requirements.txt

# download the model cache
HF_HUB_OFFLINE=0 conda run -n costume-mcp python -c \
  "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"