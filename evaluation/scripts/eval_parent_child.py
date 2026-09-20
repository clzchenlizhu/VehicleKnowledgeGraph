# -*- coding: utf-8 -*-
"""
父子块（Parent-Child / Small-to-Big）召回评测脚本
======================================================================
思路：
  - 向量库里索引的是【子块】（小块，检索粒度）；
  - 每个子块通过 parent_key 指向它的【父块】（大块，喂给 LLM 的粒度）；
  - 检索 Top-K 子块后，按 parent_key 去重，保留每个父块【首次出现】的排名；
  - 分别在【子块粒度】和【父块粒度】上算 Recall / MRR。

两层指标含义：
  - ChildRecall@k / ChildMRR：正确子块有没有进 Top-K、排第几 —— 衡量"小块检索灵不灵敏"。
  - ParentRecall@p / ParentMRR：正确父块（去重后）有没有进前 p 个、排第几 ——
    这才是 LLM 真正看到的，直接对应最终答案质量。

命中判定（与 eval_recall.py 完全一致）：
  按文本判定，ground_truth_text（去空白后）必须是子块 content 的子串才算命中。
  父块命中 = 它名下任意一个子块命中（父块是子块的超集，子块命中则父块必然包含答案）。

公平性：向量化、混合检索权重(0.9/0.1)、字段都与生产 node_search_embedding 一致。

运行（项目根目录）：
    .venv/Scripts/python evaluation/scripts/eval_parent_child.py
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
# 存【子块】的集合（父子块方案下，索引层是子块）
CHILD_COLLECTION = "kb_chunks"

TOP_K = 10          # 子块检索 Top-K（ChildRecall@10）
PARENT_TOP_P = 5   # 去重后保留的父块数（ParentRecall@5，喂给 LLM 的父块上限）

# 父块标识字段优先级：parent_id 最准；没有就退到 parent_title；再没有退到 title
PARENT_KEY_FIELDS = ["parent_id", "parent_title", "title"]

TEST_SET_PATH = SCRIPT_DIR.parent / "test_sets" / "test.json"
REPORT_PATH = SCRIPT_DIR.parent / "eval_parent_child_report.json"
# ======================================================


# ---------------- 统一召回函数（与生产一致） ----------------
def retrieve(client, collection_name: str, query: str, top_k: int = TOP_K):
    """返回命中子块列表 [{id, content, title, parent_key}, ...]。"""
    dense = generate_embeddings([query])["dense"][0]
    sparse = generate_embeddings([query])["sparse"][0]

    reqs = create_hybrid_search_requests(dense, sparse, expr=None, limit=top_k)
    resp = hybrid_search(
        client=client,
        collection_name=collection_name,
        reqs=reqs,
        ranker_weights=(0.9, 0.1),
        norm_score=True,
        limit=top_k,
        output_fields=["chunk_id", "item_name", "content", "title",
                       "parent_title", "parent_id", "part", "file_title"],
    )
    hits = resp[0] if resp else []

    parsed = []
    for h in hits:
        entity = h.get("entity", {}) or {}
        # 选第一个非空的父块标识
        parent_key = None
        for f in PARENT_KEY_FIELDS:
            v = entity.get(f)
            if v not in (None, ""):
                parent_key = str(v)
                break
        parsed.append({
            "id": h.get("id") or entity.get("chunk_id"),
            "content": entity.get("content", ""),
            "title": entity.get("title", ""),
            "parent_key": parent_key or "(无父块标识)",
        })
    return parsed


# ---------------- 命中判定（文本，与 eval_recall 一致） ----------------
def _normalize(s: str) -> str:
    return "".join((s or "").split())


def is_relevant(child_content: str, ground_truth_text: str) -> bool:
    gt = _normalize(ground_truth_text)
    cc = _normalize(child_content)
    if not gt:
        return False
    return gt in cc


def overlap_ratio(child_content: str, ground_truth_text: str) -> float:
    gt = _normalize(ground_truth_text)
    cc = _normalize(child_content)
    if not gt:
        return 0.0
    return sum(1 for ch in gt if ch in cc) / len(gt)


# ---------------- 父子块评估核心 ----------------
def evaluate_queries(client, collection_name: str, test_set: list,
                      top_k: int = TOP_K, parent_p: int = PARENT_TOP_P) -> dict:
    per_query = []

    # 子块粒度
    child_recall_hits = 0
    child_rr_sum = 0.0
    # 父块粒度
    parent_recall_hits = 0
    parent_rr_sum = 0.0

    for item in test_set:
        qid = item["query_id"]
        query = item["query"]
        gt_text = item["ground_truth_text"]

        hits = retrieve(client, collection_name, query, top_k=top_k)

        # ---- 1. 子块粒度 ----
        child_relevant_ranks = [r for r, h in enumerate(hits, start=1)
                                if is_relevant(h["content"], gt_text)]
        child_hit = 1 if child_relevant_ranks else 0
        child_rr = (1.0 / child_relevant_ranks[0]) if child_relevant_ranks else 0.0
        best_overlap = max((overlap_ratio(h["content"], gt_text) for h in hits), default=0.0)

        # ---- 2. 父块粒度：按 parent_key 去重，保留首次出现顺序 ----
        parent_order = []          # 去重后的父块序列（按首次出现）
        parent_relevant = {}      # parent_key -> 是否命中（名下有相关子块）
        for h in hits:
            pk = h["parent_key"]
            if pk not in parent_relevant:
                parent_order.append(pk)
                parent_relevant[pk] = False
            if is_relevant(h["content"], gt_text):
                parent_relevant[pk] = True

        # 前 parent_p 个父块里有没有"正确父块"
        top_parents = parent_order[:parent_p]
        parent_hit = 1 if any(parent_relevant[p] for p in top_parents) else 0
        # 第一个正确父块的排名（1 起）
        first_parent_rank = None
        for idx, pk in enumerate(parent_order, start=1):
            if parent_relevant[pk]:
                first_parent_rank = idx
                break
        parent_rr = (1.0 / first_parent_rank) if first_parent_rank else 0.0

        child_recall_hits += child_hit
        child_rr_sum += child_rr
        parent_recall_hits += parent_hit
        parent_rr_sum += parent_rr

        per_query.append({
            "query_id": qid,
            "query": query,
            "best_overlap": round(best_overlap, 2),
            # 子块层
            "child_hit": bool(child_hit),
            "child_rank": child_relevant_ranks[0] if child_relevant_ranks else None,
            # 父块层
            "parent_hit": bool(parent_hit),
            "parent_rank": first_parent_rank,
            "distinct_parents": len(parent_order),
            "top_parents": top_parents,
        })

    n = len(test_set)
    return {
        "collection": collection_name,
        "n_queries": n,
        "top_k_child": top_k,
        "top_p_parent": parent_p,
        # 子块粒度
        "child_recall@k": child_recall_hits / n if n else 0.0,
        "child_mrr": child_rr_sum / n if n else 0.0,
        # 父块粒度
        "parent_recall@p": parent_recall_hits / n if n else 0.0,
        "parent_mrr": parent_rr_sum / n if n else 0.0,
        "per_query": per_query,
    }


# ---------------- 报告 ----------------
def print_report(result: dict):
    print("\n" + "=" * 64)
    print("父子块召回评估报告（检索=子块，喂LLM=父块）")
    print("=" * 64)
    print(f"测试题数: {result['n_queries']}   子块Top-K={result['top_k_child']}   父块Top-P={result['top_p_parent']}")
    print("-" * 64)
    print(f"ChildRecall@{result['top_k_child']:<3}  {result['child_recall@k']*100:6.2f}%   （小块找得回吗）")
    print(f"ChildMRR      {result['child_mrr']*100:6.2f}%   （小块排第几）")
    print(f"ParentRecall@{result['top_p_parent']:<3}  {result['parent_recall@p']*100:6.2f}%   （喂给LLM的父块找得回吗）")
    print(f"ParentMRR     {result['parent_mrr']*100:6.2f}%   （喂给LLM的父块排第几）")
    print("-" * 64)
    print("逐题明细（✓命中 / ✗未命中）：")
    for q in result["per_query"]:
        cf = "✓" if q["child_hit"] else "✗"
        pf = "✓" if q["parent_hit"] else "✗"
        print(f"  {q['query_id']} 子块{cf} rank={q['child_rank']} | 父块{pf} rank={q['parent_rank']} "
              f"| {q['query']}")
    print("=" * 64)


def main():
    client = get_milvus_client()
    if client is None:
        raise RuntimeError("Milvus 客户端初始化失败，请检查 .env 里的 MILVUS_URL")

    test_set = json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))
    logger.info(f"加载测试集 {len(test_set)} 条：{TEST_SET_PATH}")

    result = evaluate_queries(client, CHILD_COLLECTION, test_set,
                              top_k=TOP_K, parent_p=PARENT_TOP_P)

    print_report(result)

    REPORT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入：{REPORT_PATH}")


if __name__ == "__main__":
    main()
