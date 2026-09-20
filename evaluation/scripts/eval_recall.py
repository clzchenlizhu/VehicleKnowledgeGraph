# -*- coding: utf-8 -*-
"""
切块策略召回评测脚本（对比实验）
======================================================================
用途：同一套测试题，分别在【正式语义切块集合】和【对照组固定长度集合】上检索，
      计算 Recall@k / Precision@k / MRR，量化两种切分策略的真实差距。

公平性约束（与生产检索 node_search_embedding 完全一致）：
  - 查询向量化：generate_embeddings([query])，仅用 query 本身（生产用 rewritten_query，
    这里两个集合都用同一个原始 query，保持变量一致）；
  - 混合检索：create_hybrid_search_requests + hybrid_search(ranker_weights=(0.9,0.1),
    norm_score=True)，稠密0.9 / 稀疏0.1，IP 度量；
  - 两个集合用【完全相同】的 top_k、权重、字段，唯一变量仍是切分策略。

命中判定说明：
  正式库 chunk_id 是 INT64 自增主键（auto_id），test.json 里的 ground_truth_chunk_ids
  是 chunk_xxxx 形式的逻辑 id，二者无法数值相等。因此命中按【文本】判定：
  召回块 content 是否覆盖 ground_truth_text（先精确包含，再做字符重叠兜底）。

运行（项目根目录）：
    .venv/Scripts/python evaluation/scripts/eval_recall.py
"""

import json
import sys
from pathlib import Path

# ---------------- 路径与环境 ----------------
SCRIPT_DIR = Path(__file__).resolve().parent          # evaluation/scripts
PROJECT_ROOT = SCRIPT_DIR.parent.parent                # 项目根
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import (
    get_milvus_client,
    create_hybrid_search_requests,
    hybrid_search,
)
from app.core.logger import logger

# ====================== 评测配置 ======================
# 正式语义切块集合（生产 CHUNKS_COLLECTION=kb_chunks）；对照组由 gen_fixed_chunks.py 生成
SEMANTIC_COLLECTION = "kb_chunks"
CONTROL_COLLECTION = "collection_fixed"

TOP_K = 10                 # 评测 Recall@10 / Precision@10
TEST_SET_PATH = SCRIPT_DIR.parent / "test_sets" / "test.json"
REPORT_PATH = SCRIPT_DIR.parent / "eval_report.json"

# 生产检索会按 item_name in [...] 过滤；测试题与具体商品无关，为隔离切块变量，两边都不过滤
USE_ITEM_NAME_FILTER = False
# ======================================================


# ---------------- 统一召回函数（参数与生产一致） ----------------
def retrieve(client, collection_name: str, query: str, top_k: int = TOP_K):
    """
    在指定集合上做一次混合检索，返回命中列表 [{"id","distance","content"}, ...]。
    权重/归一化/字段与 node_search_embedding.step_3 保持一致。
    """
    # 1. query 向量化
    dense, sparse = generate_embeddings([query])["dense"][0], generate_embeddings([query])["sparse"][0]

    # 2. 构造检索请求（expr=None 即不做 item_name 过滤）
    reqs = create_hybrid_search_requests(dense, sparse, expr=None, limit=top_k)

    # 3. 混合检索：稠密0.9 / 稀疏0.1，归一化评分
    resp = hybrid_search(
        client=client,
        collection_name=collection_name,
        reqs=reqs,
        ranker_weights=(0.9, 0.1),
        norm_score=True,
        limit=top_k,
        output_fields=["chunk_id", "item_name", "content", "title", "parent_title", "part", "file_title"],
    )
    hits = resp[0] if resp else []

    # 4. 统一解析成 [{id, distance, content}]，兼容 entity 包裹结构
    parsed = []
    for h in hits:
        entity = h.get("entity", {}) or {}
        parsed.append({
            "id": h.get("id") or entity.get("chunk_id"),
            "distance": h.get("distance"),
            "content": entity.get("content", ""),
            "title": entity.get("title", ""),
        })
    return parsed


# ---------------- 命中判定（文本） ----------------
def _normalize(s: str) -> str:
    """去掉所有空白，便于跨换行/空格做包含判断。"""
    return "".join((s or "").split())


def is_relevant(chunk_content: str, ground_truth_text: str) -> bool:
    """
    判定召回块是否【真正】命中标准答案：
    标准答案（去空白后）必须是块内容的子串 —— 精确包含才算命中。
    （早期版本用 0.6 字符重叠兜底，会把语义相近但原句不存在的块误判为命中，
      产生假阳性，已废弃。接近命中但不精确的情况由 report_near_miss 单独打印。）
    """
    gt = _normalize(ground_truth_text)
    cc = _normalize(chunk_content)
    if not gt:
        return False
    return gt in cc


def overlap_ratio(chunk_content: str, ground_truth_text: str) -> float:
    """标准答案字符落在块内容中的比例，仅用于排查 near-miss，不计入命中。"""
    gt = _normalize(ground_truth_text)
    cc = _normalize(chunk_content)
    if not gt:
        return 0.0
    return sum(1 for ch in gt if ch in cc) / len(gt)


# ---------------- 指标计算 ----------------
def evaluate_queries(client, collection_name: str, test_set: list, top_k: int = TOP_K) -> dict:
    """对一个集合跑完全部测试题，返回 per-query 明细 + 汇总指标。"""
    per_query = []
    recall_hits = 0
    precision_sum = 0.0
    rr_sum = 0.0

    for item in test_set:
        qid = item["query_id"]
        query = item["query"]
        gt_text = item["ground_truth_text"]

        hits = retrieve(client, collection_name, query, top_k=top_k)
        relevant_ranks = [rank for rank, h in enumerate(hits, start=1) if is_relevant(h["content"], gt_text)]
        # 仅用于排查：top_k 里与标准答案最接近的重叠度（不精确命中时看）
        best_overlap = max((overlap_ratio(h["content"], gt_text) for h in hits), default=0.0)

        hit = 1 if relevant_ranks else 0
        prec = len(relevant_ranks) / top_k
        rr = (1.0 / relevant_ranks[0]) if relevant_ranks else 0.0

        recall_hits += hit
        precision_sum += prec
        rr_sum += rr

        per_query.append({
            "query_id": qid,
            "query": query,
            "hit": bool(hit),
            "first_relevant_rank": relevant_ranks[0] if relevant_ranks else None,
            "relevant_count": len(relevant_ranks),
            "best_overlap": round(best_overlap, 2),
            "top_titles": [h["title"] for h in hits[:3]],
        })

    n = len(test_set)
    return {
        "collection": collection_name,
        "n_queries": n,
        "recall@k": recall_hits / n if n else 0.0,
        "precision@k": precision_sum / n if n else 0.0,
        "mrr": rr_sum / n if n else 0.0,
        "per_query": per_query,
    }


# ---------------- 报告 ----------------
def print_report(semantic_result: dict, control_result: dict):
    def line(name, a, b):
        return f"{name:12s}  语义={a*100:6.2f}%   固定={b*100:6.2f}%   差值={ (a-b)*100:+6.2f}pp"

    print("\n" + "=" * 60)
    print("切块策略召回对比报告")
    print("=" * 60)
    print(f"测试题数: {semantic_result['n_queries']}    Top-K: {TOP_K}")
    print(line("Recall@%d" % TOP_K, semantic_result["recall@k"], control_result["recall@k"]))
    print(line("Precision@%d" % TOP_K, semantic_result["precision@k"], control_result["precision@k"]))
    print(line("MRR", semantic_result["mrr"], control_result["mrr"]))
    print("-" * 60)
    print("语义集合逐题：")
    for q in semantic_result["per_query"]:
        flag = "✓" if q["hit"] else "✗"
        print(f"  {flag} {q['query_id']} rank={q['first_relevant_rank']} 最高重叠={q.get('best_overlap')} | {q['query']}")
    print("固定集合逐题：")
    for q in control_result["per_query"]:
        flag = "✓" if q["hit"] else "✗"
        print(f"  {flag} {q['query_id']} rank={q['first_relevant_rank']} 最高重叠={q.get('best_overlap')} | {q['query']}")
    print("=" * 60)


def main():
    client = get_milvus_client()
    if client is None:
        raise RuntimeError("Milvus 客户端初始化失败，请检查 .env 里的 MILVUS_URL")

    test_set = json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))
    logger.info(f"加载测试集 {len(test_set)} 条：{TEST_SET_PATH}")

    # 两个集合跑同一套题
    semantic_result = evaluate_queries(client, SEMANTIC_COLLECTION, test_set, top_k=TOP_K)
    control_result = evaluate_queries(client, CONTROL_COLLECTION, test_set, top_k=TOP_K)

    print_report(semantic_result, control_result)

    report = {
        "top_k": TOP_K,
        "semantic": semantic_result,
        "fixed": control_result,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入：{REPORT_PATH}")


if __name__ == "__main__":
    main()
