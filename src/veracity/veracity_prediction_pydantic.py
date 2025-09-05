import argparse
import json
import time
from tqdm.auto import tqdm
import openai
import tiktoken
import re
import sys
import os
import requests
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
from huggingface_hub import InferenceApi
import time
import datetime
from pydantic import BaseModel, AnyUrl, Field, ValidationError
from typing import List, Literal

# project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)



from factchecking_api import stance_prediction, veracity_prediction
from src.api_calls.api_manager import ApiManager


# Login HF
login("YOUR HUGGINFACE KEY HERE")
am = ApiManager()

with open("src/veracity/config.json", "r") as f:
    config = json.load(f)

def load_dataset(file_path, dataset_type):
    if dataset_type == "averitec" or dataset_type == "maldita":
        examples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        examples.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        print(f"Error decoding line:\n{line}\n{e}")
        return examples
    #elif dataset_type == "maldita":
    #    with open(file_path, "r", encoding="utf-8") as f:
    #        return json.load(f)
    else:
        raise ValueError("Invalid dataset type")

def classify_claims_huggingface(dataset, model, dataset_type, output_file=None, pipeline=None):
    results = []
    claim_items = dataset #if dataset_type == "averitec" else list(dataset.items())
    am.reset_cost()

    if dataset_type == "averitec":
        dataset_language = "en"
    elif dataset_type == "maldita":
        dataset_language = "es"


    with tqdm(total=len(claim_items)) as pbar:
        for data in claim_items:
            claim_id = data["claim_id"] #if dataset_type == "averitec" else data[0]
            claim = data["claim"] #if dataset_type == "averitec" else data[1]["claim"]
            print(f"Processing Claim ID: {claim_id}, Claim: {claim}")
            evidence_list = data["evidence"] #if dataset_type == "averitec" else data[1]["context"]

            stance_predictions = []
            questions = []
            evidences = []
            stances = []

            for ev in evidence_list:
                question = ev.get("question", "N/A")
                answer = ev.get("answer", ev.get("text", ""))
                url = ev.get("url", ev.get("metadata", {}).get("url", "N/A"))

                try:
                    stance_pred = stance_prediction(am, claim, answer, model, language=dataset_language, fewshot=args.few_shot, cot=args.cot, pipeline=pipeline)
                    if stance_pred.stance not in ["Positive", "Negative", "Neutral"]:
                        print(f"Unrecognized stance '{stance_pred.stance}'")
                        stance_pred.stance = "Neutral"  # Default to Neutral if stance is not recognized
                    
                except Exception as e:
                    print(f"[Stance Error] {claim_id}: {e}")
                    continue

                stance_predictions.append(stance_pred)
                stances.append(stance_pred.stance)
                #print(f"Stance Prediction: {stance_pred['stance']} {type(stance_pred)}") #.stance}, Reasoning: {stance_pred.reasoning}")
                print(f"Stance Prediction: {stance_pred.stance}, Reasoning: {stance_pred.reasoning}")
                questions.append(question)
                evidences.append(answer)

            try:
                veracity_pred = veracity_prediction(am, claim, questions, evidences, stances, model, language=dataset_language, dataset_type=dataset_type, fewshot=args.few_shot, cot=args.cot, useStance=args.useStance, pipeline=pipeline)
                if dataset_type == "averitec":
                    if veracity_pred.pred_label not in ["Supported", "Refuted", "Not Enough Evidence", "Conflicting/Cherrypicking"]:
                        print(f"Unrecognized veracity label '{veracity_pred.pred_label}'")
                        veracity_pred.pred_label = "Not Enough Evidence"

                elif dataset_type == "maldita":
                    if veracity_pred.pred_label not in ["Supported", "Refuted", "Not Enough Evidence"]:
                        print(f"Unrecognized veracity label '{veracity_pred.pred_label}'")
                        veracity_pred.pred_label = "Not Enough Evidence"


                #print("Veracity Prediction: ", veracity_pred['pred_label'], type(stance_pred)) #.pred_label, ", Reasoning: ", veracity_pred.reasoning)
                print("Veracity Prediction: ", veracity_pred.pred_label, ", Reasoning: ", veracity_pred.reasoning)
            except Exception as e:
                print(f"[Veracity Error] {claim_id}: {e}")
                continue

            try:
                if len(evidence_list) != len(stance_predictions):
                        print(f"[Warning] Evidence and stance prediction count mismatch for claim {claim_id}")
                        continue
                
                if "gpt" in model.lower():
                
                    structured_result = {
                        "claim_id": claim_id,
                        "claim": claim,
                        "pred_label": veracity_pred["pred_label"],
                        "reasoning": veracity_pred["reasoning"],
                        "evidence": [
                            {
                                "question": ev.get("question", ""),
                                "answer": ev.get("answer", ev.get("text", "")),
                                "url": ev.get("url", ev.get("metadata", {}).get("url", "")),
                                "reasoning": sp["reasoning"],
                                "stance": sp["stance"]
                            }
                            for ev, sp in zip(evidence_list, stance_predictions)
                            #for sp in stance_predictions
                        ]
                    }
                else:    
                    '''
                    if len(evidence_list) != len(stance_predictions):
                        print(f"[Warning] Evidence and stance prediction count mismatch for claim {claim_id}")
                        continue
                    '''

                    structured_result = {
                        "claim_id": claim_id,
                        "claim": claim,
                        #"pred_label": veracity_pred["pred_label"],
                        #"reasoning": veracity_pred["reasoning"],
                        "pred_label": veracity_pred.pred_label,
                        "reasoning": veracity_pred.reasoning,
                        "evidence": [
                            {
                                "question": ev.get("question", ""),
                                "answer": ev.get("answer", ev.get("text", "")),
                                "url": ev.get("url", ev.get("metadata", {}).get("url", "")),
                                #"stance": sp["stance"],
                                #"reasoning": sp["reasoning"]
                                "stance": sp.stance,
                                "reasoning": sp.reasoning
                            }
                            for ev, sp in zip(evidence_list, stance_predictions)
                            #for sp in stance_predictions
                        ]
                    }
                

                results.append(structured_result)
                if output_file:
                    save_results(results, output_file)
            except Exception as e:
                print(f"[Structuring Error] {claim_id}: {e}")
            pbar.update(1)
    return results



def classify_claims_openai(dataset, model, dataset_type, output_file=None):
    results = []
    claim_items = dataset #if dataset_type == "averitec" else list(dataset.items())
    am.reset_cost()
    if dataset_type == "averitec":
        dataset_language = "en"
    elif dataset_type == "maldita":
        dataset_language = "es"

    with tqdm(total=len(claim_items)) as pbar:
        for data in claim_items:
            claim_id = data["claim_id"] #if dataset_type == "averitec" else data[0]
            claim = data["claim"] #if dataset_type == "averitec" else data[1]["claim"]
            print(f"Processing Claim ID: {claim_id}, Claim: {claim}")
            evidence_list = data["evidence"] #if dataset_type == "averitec" else data[1]["context"]

            stance_predictions = []
            questions = []
            evidences = []
            stances = []

            for ev in evidence_list:
                question = ev.get("question", "N/A")
                answer = ev.get("answer", ev.get("text", ""))
                url = ev.get("url", ev.get("metadata", {}).get("url", "N/A"))

                try:
                    stance_pred = stance_prediction(am, claim, answer, model, language=dataset_language, fewshot=args.few_shot, cot=args.cot) #, pipeline=pipeline)
                    print(f"Stance Prediction: {stance_pred.stance}, Reasoning: {stance_pred.reasoning}")
                    if stance_pred.stance not in ["Positive", "Negative", "Neutral"]:
                        print(f"Unrecognized stance '{stance_pred.stance}'")
                        stance_pred.stance = "Neutral"
                except Exception as e:
                    print(f"[Stance Error] {claim_id}: {e}")
                    
                    continue

                stance_predictions.append(stance_pred)
                stances.append(stance_pred.stance)
                print(f"Stance Prediction: {stance_pred.stance} {type(stance_pred)}") #.stance}, Reasoning: {stance_pred.reasoning}")
                questions.append(question)
                evidences.append(answer)

            try:
                veracity_pred = veracity_prediction(am, claim, questions, evidences, stances, model, language=dataset_language, dataset_type=dataset_type, fewshot=args.few_shot, cot=args.cot, useStance=args.useStance) #, pipeline=pipeline)
                print("Veracity Prediction: ", veracity_pred.pred_label, type(veracity_pred)) #.pred_label, ", Reasoning: ", veracity_pred.reasoning)
                
                if dataset_type == "averitec":
                    if veracity_pred.pred_label not in ["Supported", "Refuted", "Not Enough Evidence", "Conflicting/Cherrypicking"]:
                        print(f"Unrecognized veracity label '{veracity_pred.pred_label}'")
                        veracity_pred.pred_label = "Not Enough Evidence"

                elif dataset_type == "maldita":
                    if veracity_pred.pred_label not in ["Supported", "Refuted", "Not Enough Evidence"]:
                        print(f"Unrecognized veracity label '{veracity_pred.pred_label}'")
                        veracity_pred.pred_label = "Not Enough Evidence"
                        
            except Exception as e:
                print(f"[Veracity Error] {claim_id}: {e}")
                continue

            try:
                if len(evidence_list) != len(stance_predictions):
                    print(f"[Warning] Evidence and stance prediction count mismatch for claim {claim_id}")
                    continue

                structured_result = {
                    "claim_id": claim_id,
                    "claim": claim,
                    "pred_label": veracity_pred.pred_label,
                    "reasoning": veracity_pred.reasoning,
                    "evidence": [
                        {
                            "question": ev.get("question", ""),
                            "answer": ev.get("answer", ev.get("text", "")),
                            "url": ev.get("url", ev.get("metadata", {}).get("url", "")),
                            "reasoning": sp.reasoning,
                            "stance": sp.stance
                        }
                        for ev, sp in zip(evidence_list, stance_predictions)
                        #for sp in stance_predictions
                    ]
                }
                

                results.append(structured_result)
                if output_file:
                    print("Saving results to file {output_file}")
                    save_results(results, output_file)
            except Exception as e:
                print(f"[Structuring Error] {claim_id}: {e}")
            pbar.update(1)
    return results

def save_results(results, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", choices=["averitec", "maldita"], required=True)
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--useStance", action="store_true")
    args = parser.parse_args()

    if args.cot and args.few_shot:
        mode = "cot_fewshot"
    elif args.cot:
        mode = "cot"
    elif args.few_shot:
        mode = "few_shot"
    else:
        mode = "zero_shot"
    st = "withStance" if args.useStance else "_"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #1 base_output_dir = os.path.join("averitec_dataset", "veracity", f"run_{timestamp}")
    #1 os.makedirs(base_output_dir, exist_ok=True)

    dataset_dirs = {"averitec": "averitec_dataset", "maldita": "maldita_dataset"}
    dataset_files = {
        "averitec": ["dev_top_3_rerank_qa.json"],
        "maldita": ["unique_claims_qs_context_50_formatted_per_line_TOTAL.json"] 
    }

    dataset_dir = dataset_dirs[args.dataset_type]
    
    base_output_dir = os.path.join(dataset_dir, "veracity", f"run_{timestamp}") #1
    os.makedirs(base_output_dir, exist_ok=True) #1
    
    

    for dataset_file in dataset_files[args.dataset_type]:
        dataset_path = os.path.join(dataset_dir, dataset_file)
        dataset = load_dataset(dataset_path, args.dataset_type)

        for model_key in config["models"].keys():
            print("\n ---------------------------- \n")
            print(f"Using model key: {model_key}")
            
            #model_id = "meta-llama/Meta-Llama-3-70B-Instruct"

            output_file = os.path.join(
                base_output_dir,
                f"{dataset_file.split('.')[0]}_{model_key}_{mode}_{st}_veracity_predictions.json"
            )
            if "gpt" in model_key.lower():
                print("Using OpenAI API for classification.")
                classify_claims = classify_claims_openai

                results = classify_claims(
                dataset,
                model_key,
                args.dataset_type,
                output_file=output_file,
            )
            else:
                print("Using Hugging Face API for classification.")
                model_name = config["models"][model_key]
                pipeline = transformers.pipeline(
                    "text-generation",
                    model=model_name,
                    model_kwargs={"torch_dtype": torch.bfloat16},
                    device_map="auto",
                )
                classify_claims = classify_claims_huggingface

                results = classify_claims(
                dataset,
                model_key,
                args.dataset_type,
                output_file=output_file,
                pipeline=pipeline
            )

            '''    
            results = classify_claims(
                dataset,
                model_key,
                args.dataset_type,
                output_file=output_file,
                pipeline=pipeline if "gpt" not in model_key.lower() else None
            )'''
            print(f"Finished {model_key} in {mode} mode.")
