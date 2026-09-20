# -*- coding: utf-8 -*-
"""
Rerank 重排序效果评估（复刻生产 node_rerank）
======================================================================
业务代码不变，只在评估侧衡量：CrossEncoder 重排到底有没有把正确答案往前推。

复刻生产 node_rerank 的关键步骤：
  1. 候选：用 eval_recall_rrf.retrieve_hybrid（路线A+B RRF）取 top CANDIDATE_K 条
  2. 统一成 [{title,text,type:milvus}]
  3. get_reranker_model().compute_score([[query,text],...], normalize=True) 打分
  4. 按分数降序，取 top TOP_K
  5. 对比：重排前(RRF顺序) vs 重排后(CrossEncoder顺序) 的 Recall@K / MRR

说明：
  - MCP 联网路 web_search_docs 这 31 题答案都在本地手册，联网结果不会命中 GT，
    且依赖外部服务；这里默认不传 web（= 只评 Milvus 这路被重排的效果）。
    如需接入 MCP，在 main 里把 web_docs 并入 candidates 即可，type 标成 web。
  - 同一套 test.json、同一命中判定，唯一变量 = 是否过 CrossEncoder 重排。

运行（项目根目录）：
    .venv/Scripts/python evaluation/scripts/eval_rerank.py
"""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from app.clients.milvus_utils import get_milvus_client
from app.lm.reranker_utils import get_reranker_model
from app.core.logger import logger

sys.path.insert(0, str(SCRIPT_DIR))
from eval_recall import is_relevant, overlap_ratio, TEST_SET_PATH
from eval_recall_rrf import COLLECTION, retrieve_hybrid

TOP_K = 10
CANDIDATE_K = 20          # 重排前每路取多少候选进 CrossEncoder
REPORT_PATH = SCRIPT_DIR.parent / "eval_report_rerank.json"


def rerank_candidates(query: str, candidates: list):
    """复刻 node_rerank.step_3：CrossEncoder 对 [query,text] 打分并重排。"""
    if not candidates:
        return []
    pairs = [[query, c.get("content", "") or c.get("text", "") or ""] for c in candidates]
    reranker = get_reranker_model()
    scores = reranker.compute_score(pairs, normalize=True)
    if not isinstance(scores, list):
        scores = [scores]
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    out = []
    for score, c in ranked:
        item = dict(c)
        item["rerank_score"] = round(float(score), 4)
        out.append(item)
    return out


def first_relevant_rank(hits, gt):
    for r, h in enumerate(hits, start=1):
        text = h.get("content") or h.get("text") or ""
        if is_relevant(text, gt):
            return r
    return None


def main():
    client = get_milvus_client()
    if client is None:
        raise RuntimeError("Milvus 客户端初始化失败")
    test_set = json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))
    logger.info(f"加载测试集 {len(test_set)} 条：{TEST_SET_PATH}")

    before_rr, after_rr = 0.0, 0.0
    before_hit = after_hit = 0
    detail = []

    for item in test_set:
        qid, query, gt = item["query_id"], item["query"], item["ground_truth_text"]

        # 1. RRF 粗排候选
        candidates = retrieve_hybrid(client, COLLECTION, query, top_k=CANDIDATE_K)
        # 2. CrossEncoder 重排
        reranked = rerank_candidates(query, candidates)

        # 3. 指标：重排前(RRF顺序，取topK) vs 重排后(取topK)
        rrf_hits = candidates[:TOP_K]
        re_hits = reranked[:TOP_K]
        rb = first_relevant_rank(rrf_hits, gt)
        ra = first_relevant_rank(re_hits, gt)

        before_rr += (1.0 / rb) if rb else 0.0
        after_rr += (1.0 / ra) if ra else 0.0
        before_hit += 1 if rb else 0
        after_hit += 1 if ra else 0

        detail.append({
            "query_id": qid, "query": query,
            "rrf_rank": rb, "rerank_rank": ra,
            "top_after": [
                {"rank": i, "score": h.get("rerank_score"),
                 "title": h.get("title", "")}
                for i, h in enumerate(re_hits[:3], start=1)
            ],
        })

    n = len(test_set)
    report = {
        "collection": COLLECTION, "top_k": TOP_K, "candidate_k": CANDIDATE_K,
        "before_rerank": {
            "recall@k": before_hit / n, "mrr": before_rr / n,
        },
        "after_rerank": {
            "recall@k": after_hit / n, "mrr": after_rr / n,
        },
        "per_query": detail,
    }

    print("\n" + "=" * 60)
    print("RRF粗排 vs CrossEncoder重排 对比")
    print("=" * 60)
    print(f"测试题数: {n}   Top-K: {TOP_K}   候选: {CANDIDATE_K}")
    print(f"Recall@10  重排前={before_hit/n*100:6.2f}%   重排后={after_hit/n*100:6.2f}%   "
          f"差值={(after_hit-before_hit)/n*100:+6.2f}pp")
    print(f"MRR        重排前={before_rr/n:6.4f}   重排后={after_rr/n:6.4f}   "
          f"差值={(after_rr-before_rr)/n:+6.4f}")
    print("-" * 60)
    print("逐题排名变化（RRF -> 重排后）：")
    for d in detail:
        flag = "✓" if d["rerank_rank"] else "✗"
        print(f"  {flag} {d['query_id']:>5} RRF={d['rrf_rank']} 重排={d['rerank_rank']} | {d['query']}")
    print("=" * 60)

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入：{REPORT_PATH}")


if __name__ == "__main__":
    main()
