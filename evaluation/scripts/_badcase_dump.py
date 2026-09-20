# -*- coding: utf-8 -*-
"""导出 Bad Case 的完整 Top10 明细，供根因分析。"""
import sys
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))  # 让本目录可 import eval_recall

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from eval_recall import retrieve, is_relevant, SEMANTIC_COLLECTION, CONTROL_COLLECTION, TOP_K, TEST_SET_PATH
from app.clients.milvus_utils import get_milvus_client

# 固定排名>=5 或 任一集合完全未命中
BAD_QIDS = {"q06", "q08", "q18", "q19", "q22", "q29", "q36"}

test_set = json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))
client = get_milvus_client()

out_dir = SCRIPT_DIR.parent / "bad_cases/chunk_split"
out_dir.mkdir(exist_ok=True)

result = {}
for item in test_set:
    qid = item["query_id"]
    if qid not in BAD_QIDS:
        continue
    q = item["query"]
    gt = item["ground_truth_text"]
    entry = {"query": q, "ground_truth": gt, "semantic": {}, "fixed": {}}
    for key, col in [("semantic", SEMANTIC_COLLECTION), ("fixed", CONTROL_COLLECTION)]:
        hits = retrieve(client, col, q, top_k=TOP_K)
        rank = next((i for i, h in enumerate(hits, start=1) if is_relevant(h["content"], gt)), None)
        top10 = [
            {
                "rank": i,
                "id": h["id"],
                "title": h["title"],
                "content": (h["content"] or "")[:160],
            }
            for i, h in enumerate(hits, start=1)
        ]
        entry[key] = {"rank": rank, "top10": top10}
    result[qid] = entry
    print(f"{qid}: semantic_rank={entry['semantic']['rank']} fixed_rank={entry['fixed']['rank']}")

(out_dir / "_raw.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print("saved", out_dir / "_raw.json")
