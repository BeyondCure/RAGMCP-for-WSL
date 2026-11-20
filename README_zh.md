# RAGMCP-for-WSL

<p align="center">
  🌐&nbsp; <a href="README.md">English</a> &nbsp;|&nbsp; <strong>中文</strong>
</p>

一个本地 RAG + MCP 服务：把提供的知识库（markdown文档、text文档等）建立向量索引，通过 MCP 协议对外提供检索工具。默认在 WSL 中以 streamable-HTTP 常驻，Windows 和 WSL 的 MCP 客户端都可以连接。

更准确地说，本项目是一个演示框架/模板，旨在帮助大家理解一个 RAG + MCP 服务的整体结构；实际使用时，通常需要针对自己的语料和需求做一定的自定义或二次开发。

## 快速开始

### 1. 创建环境

```bash
conda create -n costume-mcp python=3.12 -y
conda activate costume-mcp

# 纯 CPU 环境建议先装 CPU 版 torch，避免拉取 CUDA 依赖
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. 配置

```bash
cp .env.example .env
```

实际配置改 `.env`；`config.py` 只定义配置项和默认值，不需要修改。环境变量优先级高于 `.env`，例如 `MCP_PORT=9000 python runMCPServer.py`。

### 3. 首次下载模型

模型约 2.2GB，下载一次即可；之后保持 `HF_HUB_OFFLINE=true`：

```bash
HF_HUB_OFFLINE=0 python -c \
  "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
```

### 4. 准备文档并建索引

仓库不包含文档和索引。把自己的 Markdown/PDF/DOCX 放进 `data/raw/`，然后：

```bash
python ingest.py data/raw/your-docs --collection mydocs --reset
```

之后调用 `search_knowledge` 时传 `collection="mydocs"`（或为它新增专用工具）。

### 5. 启动服务

```bash
python runMCPServer.py
# 监听 http://127.0.0.1:8000/mcp
```

测试用 stdio：

```bash
python runMCPServer.py --stdio
```

### 6. 接入客户端

Claude Code：

```bash
claude mcp add --transport http costume-rag http://localhost:8000/mcp
```

其他支持 streamable-HTTP 的 MCP 客户端，地址填 `http://localhost:8000/mcp` 即可。

Windows 访问 WSL 服务依赖 localhost 转发。如果连不上，在 `.env` 中设置 `MCP_HOST=0.0.0.0` 后重启服务。

## MCP 工具

| 工具 | 参数 | 说明 |
| --- | --- | --- |
| `search_neoforge_1_21_1_dev_docs` | `query`，`top_k=5` | 搜索 NeoForge 1.21.1 开发文档（固定 `neoforge` collection） |
| `search_knowledge` | `query`，`top_k=5`，`collection="neoforge"` | 泛用检索，可指定其他 collection，也可作为模板 |
| `list_collections` | 无 | 列出 collection、chunk 数、文档数和标题覆盖 |

`top_k` 上限由 `MCP_MAX_TOP_K=10` 控制；默认完整返回 chunk，不截断。

## 常用操作

```bash
# 查看 collection 概况（本地调试脚本，不入 git）
python costume_debug/collections_info.py

# 命令行检索看效果
python costume_debug/search_cli.py "怎么注册一个Block" -k 3

# 重建索引（服务运行时先停止服务）
python ingest.py data/raw/NeoForge-version-1.21.1 --collection neoforge --reset
```

入库语义：

- 不加 `--reset`：只处理传入的文件/目录，按文件替换旧 chunk，未传入的文件不动
- 加 `--reset`：删除整个 collection 后重建
- 改切分参数、换 embedding 模型或删除文档后，需要 `--reset`
- 服务常驻时不要入库，入库完成后重启服务

## 关键配置

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `CHROMA_DIR` | `data/index` | Chroma 持久化目录 |
| `EMBED_MODEL` | `BAAI/bge-m3` | 向量模型 |
| `HF_HUB_OFFLINE` | `true` | 只使用本地模型缓存 |
| `CHUNK_SIZE` | `900` | 目标块大小（字符） |
| `HYBRID_ENABLED` | `false` | 是否启用 dense + BM25 混合检索 |
| `RERANK_ENABLED` | `false` | 是否启用 cross-encoder 精排 |
| `MCP_TRANSPORT` | `streamable-http` | `streamable-http` / `stdio` |
| `MCP_HOST` | `127.0.0.1` | HTTP 监听地址 |
| `MCP_PORT` | `8000` | HTTP 端口 |
| `MCP_PRELOAD` | `true` | 启动时加载并 warmup 模型 |
| `MCP_MAX_TOP_K` | `10` | 检索数量上限 |
| `MCP_MAX_RESULT_CHARS` | `0` | 0 表示不截断结果 |

完整列表见 [.env.example](.env.example)。

## 目录结构

```text
CostumeMCP/
├── runMCPServer.py          # MCP 服务入口
├── ingest.py                # 离线入库 CLI
├── costume_mcp/
│   ├── config.py            # 配置定义与默认值
│   ├── mcp_server.py        # MCP 工具与校验
│   └── rag/                 # embedding、Chroma、切分、检索
├── data/
│   ├── raw/                 # 原始文档
│   └── index/               # Chroma 索引
├── .env.example             # 配置模板
├── requirements.txt         # 依赖列表
├── start_server.bash        # 启动脚本示例
└── setup.bash               # 快速配置项目脚本示例
```

## 注意事项

- 服务无鉴权，仅建议绑定本机使用。
- bge-m3 CPU 常驻约 3GB 内存；HTTP 启动 warmup 约 6 秒。
- 日志输出到 stderr；stdio 模式下不要向 stdout 打印任何内容。
