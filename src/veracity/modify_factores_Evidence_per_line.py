import json
import os
import sys

#from pydantic import BaseModel
from tqdm.auto import tqdm

# project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)


file_qs = "unique_claims_qs_context_50_2"
input_file = f"factores_dataset/{file_qs}_summarized.json"
#output_jsonl_file = f"factores_dataset/{file_qs}_formatted.jsonl"
output_jsonl_file = f"factores_dataset/{file_qs}_formatted_per_line.jsonl"

with open(input_file, "r", encoding="utf8") as f:
    raw_data = json.load(f)

with open(output_jsonl_file, "w", encoding="utf8") as f_out:
    for claim_id, content in raw_data.items():
        claim_text = content.get("claim", "").strip()
        context = content.get("context", {})

        for question, evidences in context.items():
            evidence_list = []
            if evidences:
                first_ev = evidences[0]  # Solo la primera evidencia
                evidence_list.append({
                    "question": question.strip(),
                    "metadata": first_ev.get("metadata", {}),
                    "text": first_ev.get("text", "").strip(),
                    "score": first_ev.get("score", None),
                    "answer": first_ev.get("summarized_text", "").strip()
                })

            jsonl_obj = {
                "claim_id": claim_id,
                "claim": claim_text,
                "evidence": evidence_list
            }

            f_out.write(json.dumps(jsonl_obj, ensure_ascii=False) + "\n")

print(f"JSONL file written to: {output_jsonl_file}")
