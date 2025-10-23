# Multilingual_AFC

This work introduces **FactOReS**, the first publicly available dataset for evidence-based veracity prediction in Spanish, constructed from authentic Spanish-language claims sourced from [Maldita.es](https://maldita.es), a leading Spanish fact-checking organization. We establish performance baselines by systematically applying In-Context Learning (ICL) with Large Language Models (LLMs) to both an established English dataset and our novel Spanish dataset.

This dataset contains **verifiable claims** paired with **verification questions and contextual evidence snippets** (571 instances in total) in Spanish extracted from online sources.  
It is designed for the **Automatic Fact-Checking (AFC)** task.

Each entry includes:
- `claim_id`: unique identifier of the claim  
- `claim`: textual claim to be verified  
- `question`: a question targeting aspects of the claim  
- `summarized_text`: summary of the retrieved or supporting evidence  
- `relevance`: binary indicator of evidence relevance  
- `critical_*`: critical dimensions (what, who, where, when, how) capturing key fact-checking attributes (values: 1 or 0 (null)) 
- `objectivity`: binary indicator of objectivity in the evidence  
- `TOTAL`: aggregated score  
- `STANCE`: stance of the evidence relative to the claim (`Positive`, `Negative`, `Neutral`)  
- `label`: gold veracity label for fact-checking (`Supported`, `Refuted`, `Not Enough Evidence`)  


---

## Example of an entry

```json
{
        "claim_id":6901,
        "claim":"Diario Sur tuitea que Málaga, Marbella y prácticamente toda la Costa del Sol cerrarán toda la actividad no esencial",
        "question":"Cuáles son las últimas medidas anunciadas oficialmente por el Ayuntamiento de Málaga sobre actividades no esenciales?",
        "summarized_text":"Málaga capital ha superado la tasa de mil contagios de COVID-19 por cada cien mil habitantes, lo que obliga al cierre de negocios no esenciales durante al menos 14 días a partir de este miércoles. Esta medida, establecida por la Junta de Andalucía, busca frenar la propagación del virus en sectores como la hostelería, comercio y cultura. Además, otros municipios como Casares, Ojén, Benaoján y otros también implementarán estas restricciones debido a la alta incidencia.",
        "relevance":1,
        "critical_what":1,
        "critical_who":0,
        "critical_where":1,
        "critical_when":0,
        "critical_how":0,
        "objectivity":1,
        "TOTAL":4,
        "STANCE":"Positive",
        "label":"Supported"
}
```

---


Follow these steps in order to recreate the dataset:

## 1. Generate Dataset

The first step is to generate the dataset, which involves preprocessing the original dataset, performing **Question Generation** and **Evidence Retrieval**.  
Run the `generate_dataset.ipynb` notebook and follow the steps.

- Create an environment using `requirements_AFC.txt` and activate it.  
- Make sure you have all the required API keys (**OPENAI**, **HUGGINGFACE**) with permissions for the following models:  
  - GPT-4o  
  - Qwen 2.5 (7B & 72B Instruct)  
  - LLaMA 3 (8B & 70B Instruct)  

---

To reproduce experimental results:


## 2. Perform Stance and Veracity Prediction

### 2.0. Filter Dataset  
The final JSON output from the notebook is filtered so that we only keep **50 chunks of evidence per question**.  
- **Filename:** `unique_claims_qs_context_50_filtered.json`

### 2.1. Summarize First Evidence  
For each question, only the first evidence is kept and summarized.  
- Execute: `summarize_1evidence.py`  
- **Resulting filename:** `unique_claims_qs_context_50_summarized.json`

### 2.2. Format Dataset (One Evidence per Line)  
Modify the dataset so that each line contains one **claim–question–evidence** trio.  
- Execute: `modify_maldita_Evidence_per_line.py`  
- **Resulting filename:** `unique_claims_qs_context_50_formatted_per_line.jsonl`

### 2.3. Perform Stance and Veracity Prediction  
Run stance and veracity prediction on the formatted dataset.  
(`python src/veracity/veracity_prediction_pydantic.py --dataset_type [maldita or averitec] [--few_shot] [--useStance]`) 

### 2.4. Evaluate obtained predictions: dev.json files of each dataset are the gold datasets.
- Note that this is not necessary. Intermediate steps to reproduce FactOReS dev.json, execute: 1. `create_annotation_csv.py` (from JSON format to csv), 2. Manually annotate the dataset, 3. `analyze_annotation_file.py` (check dataset distribution), 4. `calculate_agreement.py` (Inter-Annotation Agreement), 4. `xlsxs_to_json.py` (.xlsxs format to JSON).
- Execute: `stance_eval_Maldita.py` for stance evaluation with data from FactOReS dataset
- Execute: `veracity_evaluation.py` for veracity prediction with data from AVeriTeC or Maldita (`python src/veracity/veracity_evaluation.py --dataset_type [averitec_dataset or maldita_dataset]`)
   
