# Multilingual_AFC

## 1. Generate Dataset

The first step is to generate the dataset, which involves preprocessing the original dataset, performing **Question Generation** and **Evidence Retrieval**.  
Run the `generate_dataset.ipynb` notebook and follow the steps.

- Create an environment using `requirements_AFC.txt` and activate it.  
- Make sure you have all the required API keys (**OPENAI**, **HUGGINGFACE**) with permissions for the following models:  
  - GPT-4o  
  - Qwen 2.5 (7B & 72B Instruct)  
  - LLaMA 3 (8B & 70B Instruct)  

---

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
- Note that this is not necessary. Intermediate steps to reproduce Maldita dev.json, execute: 1. `create_annotation_csv.py` (from JSON format to csv), 2. Manually annotate the dataset, 3. `analyze_annotation_file.py` (check dataset distribution), 4. `calculate_agreement.py` (Inter-Annotation Agreement), 4. `xlsxs_to_json.py` (.xlsxs format to JSON).
- Execute: `stance_eval_Maldita.py` for stance evaluation with data from Maldita dataset
- Execute: `veracity_evaluation.py` for veracity prediction with data from AVeriTeC or Maldita (`python src/veracity/veracity_evaluation.py --dataset_type [averitec_dataset or maldita_dataset]`)
   
