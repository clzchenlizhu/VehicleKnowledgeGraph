# EcomGraph RAG System

基于 **LangGraph** 的企业级多路并行 RAG 智能问答系统：融合 **BGE-M3 稠密 + 稀疏混合向量检索**、**HyDE**、**Web/MCP 搜索** 与 **Neo4j 知识图谱** 四路召回，经 **RRF 融合 + CrossEncoder 重排** 后由大模型生成可溯源答案。同时支持 PDF / Word / PPT / Excel 多格式文档解析入库。



***

## ✨ 核心特性



* **多路并行召回**：向量（稠密 + 稀疏混合）、HyDE、Web/MCP、Neo4j 图谱四路并行，互补召回。

* **混合检索与精排**：Milvus 标量过滤 + 混合检索，RRF（k=60）融合，bge-reranker/CrossEncoder 精排。

* **多格式文档入库**：PDF / Word / PPT / Excel 统一解析为 Markdown，表格、公式、图片语义描述均可检索。

* **语义切块 + 父子块**：按标题层级语义初切、超长递归精切，块自带章节标题元数据。

* **知识图谱**：从 MySQL 同步商品 / 品牌 / 类目实体与关系，LLM 动态生成 Cypher，带只读安全网关与模板回退。

* **高可用兜底**：单路异常降级为空结果，不拖垮主链路；join 屏障节点避免下游重复触发。

* **可观测与评估**：全链路 Loguru + request\_id 日志；离线 Recall/Precision/MRR 评估与 LLM-as-judge 端到端打分。

## 🧱 技术栈



| 类别   | 选型                                             |
| ---- | ---------------------------------------------- |
| 编排   | LangGraph、LangChain                            |
| 向量模型 | BGE-M3（稠密 1024 维 + 稀疏）                         |
| 向量库  | Milvus（HNSW、混合检索、标量过滤）                         |
| 重排   | bge-reranker / CrossEncoder                    |
| 图谱   | Neo4j（动态 Cypher + 只读网关）                        |
| 存储   | MySQL（结构化数据）、MinIO（对象）、MongoDB（对话历史）、Redis（缓存） |
| 大模型  | 通义千问 Qwen（DashScope）                           |
| 服务   | FastAPI、Streamlit、SSE 流式输出、Docker              |

## 🗂️ 目录结构



```
app/
├── import_process/                 # 离线：知识库构建流水线（导入图）
│   └── agent/
│       ├── main_graph.py / state.py
│       └── nodes/
│           ├── node_entry.py               # 入库入口、任务分发
│           ├── node_type_recognition.py    # 文档类型识别
│           ├── node_pdf_to_md.py           # PDF → Markdown
│           ├── node_word_to_md.py          # Word → Markdown
│           ├── node_ppt_to_md.py           # PPT → Markdown
│           ├── node_md_img.py             # 图片提取与语义描述
│           ├── node_document_split.py      # 语义切块（标题初切+递归精切+重叠+元数据）
│           ├── node_item_name_recognition.py  # 主体/商品实体识别
│           ├── node_bge_embedding.py       # BGE-M3 稠密+稀疏双向量生成
│           ├── node_import_milvus.py      # 向量+元数据写入 Milvus
│           └── graph_sync/                 # 知识图谱构建
│               ├── node_init_mysql.py         # 初始化 MySQL 商品/类目数据
│               ├── node_text_ner_sync.py      # NER 抽取实体关系
│               └── node_full_sync.py           # 批量同步 Neo4j 节点与边
│
├── query_process/                 # 在线：问答检索流水线（查询图）
│   └── agent/
│       ├── main_graph.py / state.py
│       └── nodes/
│           ├── node_item_name_confirm.py  # 商品名确认/信息补全
│           ├── node_search_embedding.py   # 向量检索（稠密+稀疏混合）
│           ├── node_search_embedding_hyde.py  # HyDE 假设文档检索
│           ├── node_web_search_mcp.py    # Web/MCP 网络搜索
│           ├── node_cypher_generate.py    # LLM 生成 Cypher
│           ├── node_cypher_check.py      # 只读安全校验（拦截写操作）
│           ├── node_cypher_polish.py     # Cypher 优化
│           ├── node_cypher_execute.py    # 执行图谱查询（模板回退兜底）
│           ├── node_join.py              # 三路屏障（齐了再触发重排）
│           ├── node_rrf.py               # RRF 多路融合去重
│           ├── node_rerank.py           # CrossEncoder 精排
│           └── node_answer_output.py     # LLM 生成答案 + SSE 流式 + 引用溯源
│
├── clients/                       # 外部组件封装
│   ├── milvus_utils.py            # Milvus 连接/混合检索/写入
│   ├── minio_utils.py             # MinIO 对象存储
│   └── mongo_history_utils.py     # MongoDB 对话历史
├── lm/                            # 模型调用
│   ├── embedding_utils.py         # BGE-M3 向量化
│   ├── lm_utils.py                # Qwen 调用
│   └── reranker_utils.py         # CrossEncoder 重排
├── core/                          # 基础设施：prompt 加载、日志
├── tool/                          # 模型下载脚本（BGE-M3 / reranker）
└── utils/
evaluation/                        # 离线评估（与主业务隔离）
├── test_sets/test.json            # 标注测试集（含知识库外拒答题）
└── scripts/
    ├── eval_recall.py             # 检索层 Recall/Precision/MRR
    ├── eval_recall_rrf.py         # 纯向量 vs HyDE-RRF 对比
    ├── eval_rerank.py              # 重排效果评估
    ├── eval_parent_child.py       # 父子块评估
    ├── gen_fixed_chunks.py        # 对照组：固定长度切块
    └── badcase_dump_rrf.py        # Bad Case 明细导出
test/
└── server.py                      # 端到端批处理评估入口
prompts/                           # 各节点 Prompt 模板
```

## 🔄 核心流程

**离线入库**：原始文档（PDF/Word/PPT/Excel）→ 类型识别 → 多格式解析为 Markdown → 图片语义描述 → 语义切块 → 主体识别 → BGE-M3 双向量 → 写入 Milvus；同时 MySQL → NER → Neo4j 构建知识图谱。

**在线问答**：用户提问 → 商品名确认 → 四路并行召回（向量 / HyDE / Web-MCP / 图谱）→ join 屏障 → RRF 融合 → CrossEncoder 重排 → LLM 生成带引用答案 → SSE 流式输出。

## 🚀 快速开始



```
# 1. 安装依赖
pip install -e .        # 或 pip install -r requirements.txt

# 2. 下载模型（BGE-M3、bge-reranker）
python -m app.tool.download_bgem3
python -m app.tool.download_reranker

# 3. 配置环境变量（见 .env）
#    Milvus / Neo4j / MySQL / MinIO / DashScope API Key

# 4. 启动服务
uvicorn app.query_process.api.main:app --reload
```

## 📊 评估



```
# 检索层指标（自动剔除知识库外拒答题）
python evaluation/scripts/eval_recall.py

# 端到端批处理评估（LLM-as-judge）
python test/server.py --batch
```

关键结论（31\~100 题测试集）：语义切块与固定长度 **Recall@10 持平（≈96.8%）**，但 **MRR 语义 82.9% vs 固定 67.2%**，差距在排序；CrossEncoder 重排与父子块进一步提升 MRR；HyDE 经 A/B 实测略负向且多一次 LLM 调用，做成开关默认关闭。

## 📈 效果



* Recall@10：约 81% → 95%+

* 端到端答案准确率：约 82% → 91%+，幻觉率约 18% → 7% 以内

* P99 响应 < 200ms，SSE 首 Token < 80ms

## 📄 License

MIT