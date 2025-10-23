import json
import csv

with open('factores_dataset/unique_claims_qs_context_50_2_summarized.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

rows = []

for claim_id, content in data.items():
    claim = content["claim"]
    questions = content["questions"]
    context = content["context"]

    for question in questions:
        if question in context:
            evidences = context[question][:2]  
            for evidence in evidences:
                summarized_text = evidence.get("summarized_text", "")
                row = {
                    "claim_id": claim_id,
                    "claim": claim,
                    "question": question,
                    "summarized_text": summarized_text,
                    "relevance": "",
                    "critical_what": "",
                    "critical_who": "",
                    "critical_where": "",
                    "critical_when": "",
                    "critical_how": "",
                    "objectivity": ""
                }
                rows.append(row)

with open('factores_dataset/annotation_context_50_2_summarized.csv', 'w', encoding='utf-8', newline='') as csvfile:
    fieldnames = ["claim_id", "claim", "question", "summarized_text", "relevance", "critical_what", "critical_who", "critical_where", "critical_when", "critical_how", "objectivity"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
