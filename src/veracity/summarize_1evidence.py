import json

import json
import os
import sys

#from pydantic import BaseModel
from tqdm.auto import tqdm

# project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.api_calls.api_manager import ApiManager
from factchecking_api import summarize_evidence


os.environ["OPENAI_API_KEY"] = "YOUR OPENAI API KEY"

# Cargar datos originales
with open("maldita_dataset/unique_claims_qs_context_50_2_filtered.json", "r", encoding="utf8") as f:
    raw_data = json.load(f)

# Convertir dict a lista con 'id' incluido
data = [{"id": cid, **info} for cid, info in raw_data.items()]

if os.path.exists("maldita_dataset/unique_claims_qs_context_50_2_summarized.json"):
    if os.path.getsize("maldita_dataset/unique_claims_qs_context_50_2_summarized.json") == 0:
        print("Corrupt empty JSON file found. Deleting it.")
        os.remove("maldita_dataset/unique_claims_qs_context_50_2_summarized.json")
        annotated_data = {}
    else:
        with open("maldita_dataset/unique_claims_qs_context_50_2_summarized.json", "r", encoding="utf8") as f:
            annotated_data = json.load(f)
else:
    annotated_data = {}


print(f"Total elements: {len(data)}")
print(f"Already annotated elements: {len(annotated_data)}")

# Filtrar los que ya hemos anotado
data = [fc for fc in data if fc["id"] not in annotated_data]
print(f"Elements to annotate: {len(data)}")



am = ApiManager()
am.reset_cost()




with tqdm(data, desc="Summarizing evidences", total=len(data)) as pbar:
    for fc in data:
        claim_id = fc["id"]
        claim = fc["claim"]
        questions = fc["questions"]
        context = fc["context"]

        tqdm.write(f"\nProcessing claim ID {claim_id}: {claim}")

        # Summarize each evidence text for each question
        for question, evidences in context.items():
            for i, ev in enumerate(evidences):
                if i >= 2:
                    break  # Only summarize first two evidences
                if not ev.get("summarized_text"):
                    summary = summarize_evidence(
                        am=am,
                        top_evidence=ev["text"],
                        model="gpt4o",
                        language="es"
                    )
                    print("Summary:", summary if summary else "no summary done")
                    ev["summarized_text"] = summary if summary else ""

        # Save the updated structure
        annotated_data[claim_id] = {
            "claim": claim,
            "questions": questions,
            "context": context
        }

        # Save progress to JSON after each claim
        with open("maldita_dataset/unique_claims_qs_context_50_2_summarized.json", "w", encoding="utf8") as f:
            json.dump(annotated_data, f, indent=4, ensure_ascii=False)

        pbar.update(1)
