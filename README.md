# RAGMCP-for-WSL

<p align="center">
  🌐&nbsp; <strong>English</strong> &nbsp;|&nbsp; <a href="README_zh.md">中文</a>
</p>

A local RAG + MCP service that builds a vector index over your knowledge base (Markdown, text, and other documents) and exposes retrieval tools over the Model Context Protocol (MCP). By default it runs persistently in WSL over streamable HTTP, and MCP clients on both Windows and WSL can connect.

More precisely, this project is a demo framework/template intended to help you understand how a RAG + MCP service is put together. Using it in practice will usually require some customization or further development for your own corpus and requirements.

## Quick Start

### 1. Create the environment

```bash
conda create -n costume-mcp python=3.12 -y
conda activate costume-mcp

# CPU-only machines: install the CPU build of torch first to avoid CUDA dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` for actual configuration. `config.py` only defines the available settings and their defaults; you do not need to modify it. Environment variables take precedence over `.env`, for example `MCP_PORT=9000 python runMCPServer.py`.

### 3. Download the model (first run)

The model is about 2.2 GB and only needs to be downloaded once. Afterwards keep `HF_HUB_OFFLINE=true`:

```bash
HF_HUB_OFFLINE=0 python -c \
  "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
```

### 4. Prepare documents and build the index

The repository does not include documents or the index. Put your Markdown/PDF/DOCX files under `data/raw/`, then:

```bash
python ingest.py data/raw/your-docs --collection mydocs --reset
```

Then call `search_knowledge` with `collection="mydocs"` (or add a dedicated tool for that corpus).

### 5. Start the server

```bash
python runMCPServer.py
# Listening on http://127.0.0.1:8000/mcp
```

For testing over stdio:

```bash
python runMCPServer.py --stdio
```

### 6. Connect a client

Claude Code:

```bash
claude mcp add --transport http costume-rag http://localhost:8000/mcp
```

For other MCP clients that support streamable HTTP, use `http://localhost:8000/mcp` as the server URL.

Windows access to the WSL service relies on localhost forwarding. If it does not connect, set `MCP_HOST=0.0.0.0` in `.env` and restart the server.

## MCP Tools

| Tool | Arguments | Description |
| --- | --- | --- |
| `search_neoforge_1_21_1_dev_docs` | `query`, `top_k=5` | Search the NeoForge 1.21.1 developer docs (fixed to the `neoforge` collection) |
| `search_knowledge` | `query`, `top_k=5`, `collection="neoforge"` | Generic search across collections; useful as a template for other corpora |
| `list_collections` | None | List collections with chunk counts, document counts, and heading coverage |

`top_k` is capped by `MCP_MAX_TOP_K=10`; chunks are returned in full by default (no truncation).

## Common Operations

```bash
# Inspect collections (local debug script, not tracked in git)
python costume_debug/collections_info.py

# Try a search from the command line
python costume_debug/search_cli.py "How do I register a block?" -k 3

# Rebuild the index (stop the server first)
python ingest.py data/raw/NeoForge-version-1.21.1 --collection neoforge --reset
```

Ingestion semantics:

- Without `--reset`: only the supplied files/directories are processed; existing chunks for each file are replaced and other files are untouched.
- With `--reset`: the entire collection is dropped and rebuilt.
- After changing chunking parameters, switching embedding models, or deleting documents, use `--reset`.
- Do not ingest while the server is running; restart the server after ingestion completes.

## Key Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `CHROMA_DIR` | `data/index` | Chroma persistence directory |
| `EMBED_MODEL` | `BAAI/bge-m3` | Embedding model |
| `HF_HUB_OFFLINE` | `true` | Use the local model cache only |
| `CHUNK_SIZE` | `900` | Target chunk size (characters) |
| `HYBRID_ENABLED` | `false` | Enable dense + BM25 hybrid retrieval |
| `RERANK_ENABLED` | `false` | Enable cross-encoder reranking |
| `MCP_TRANSPORT` | `streamable-http` | `streamable-http` / `stdio` |
| `MCP_HOST` | `127.0.0.1` | HTTP bind address |
| `MCP_PORT` | `8000` | HTTP port |
| `MCP_PRELOAD` | `true` | Load and warm up the model at startup |
| `MCP_MAX_TOP_K` | `10` | Maximum number of results |
| `MCP_MAX_RESULT_CHARS` | `0` | 0 means no truncation |

See [.env.example](.env.example) for the full list.

## Project Layout

```text
CostumeMCP/
├── runMCPServer.py          # MCP server entry point
├── ingest.py                # Offline ingestion CLI
├── costume_mcp/
│   ├── config.py            # Settings and defaults
│   ├── mcp_server.py        # MCP tools and validation
│   └── rag/                 # Embedding, Chroma, chunking, retrieval
├── data/
│   ├── raw/                 # Source documents
│   └── index/               # Chroma index
├── .env.example             # Configuration template
├── requirements.txt         # Dependencies
├── start_server.bash        # Example start script
└── setup.bash               # Example environment setup script
```

## Notes

- The service has no authentication; bind it to localhost only.
- bge-m3 uses about 3 GB of RAM when resident on CPU; HTTP startup warmup takes about 6 seconds.
- Logs are written to stderr; in stdio mode, never print anything to stdout.
