
'''
def summarize_evidence_prompt(
    top_evidence: str, lang: str = "es", location: str = None
) -> str:
    return f"""
You are tasked with generating a summary for a given evidence. 

Here is the evidence to analyze:
{top_evidence}

Write a short and clear summary, using no more than 3 sentences.


Your answer should be in {'Spanish' if lang == 'es' else 'English'}.

    """.strip()

'''

def summarize_evidence_prompt(top_evidence: str, lang: str = "es", location: str = None) -> str:
    return f"""
You are tasked with generating a short and clear summary of the following evidence.

Here is the evidence:
{top_evidence}

Write a short summary in no more than 3 sentences.

Respond ONLY with a JSON object in this format:
{{
  "summarized_evidence": "your summary here"
}}

Your answer must be in {'Spanish' if lang == 'es' else 'English'}.
""".strip()

