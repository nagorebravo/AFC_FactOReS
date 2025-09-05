import os
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import sys
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from typing import Optional


# project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)


MODEL_CACHE_DIR = "./models_cache"
_loaded_models = {}
_loaded_tokenizers = {}

with open("src/veracity/config.json", "r") as f:
    config = json.load(f)


'''
def is_model_cached(model_name):
    model_dir = os.path.join(MODEL_CACHE_DIR, model_name.replace("/", "--"))
    return os.path.exists(model_dir) and os.listdir(model_dir)
'''

def is_model_cached(model_name):
    cache_name = "models--" + model_name.replace("/", "--")
    model_dir = os.path.join(MODEL_CACHE_DIR, cache_name)
    return os.path.isdir(model_dir) and bool(os.listdir(model_dir))

def get_model_config(model_key):
    entry = config["models"].get(model_key)
    if isinstance(entry, dict):
        return entry["name"], entry.get("provider", "openai")
    return entry, "openai"

def load_model_and_tokenizer(model_key):
    model_name, _ = get_model_config(model_key)
    if model_key not in _loaded_models:
        if is_model_cached(model_name):
            print(f"Model {model_key} already cached.")
        else:
            print(f"Downloading and caching model {model_key} ({model_name})...")
        

        #print(f"Loading model {model_key} ({model_name})...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=MODEL_CACHE_DIR,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        model.eval()
        _loaded_models[model_key] = model
        _loaded_tokenizers[model_key] = tokenizer
    return _loaded_models[model_key], _loaded_tokenizers[model_key]



def call_huggingface_model(model_key, prompt, temperature=0.3, max_new_tokens=300):
    model, tokenizer = load_model_and_tokenizer(model_key)
    #model, tokenizer = load_model_and_tokenizer(model_name)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)


    num_tokens = inputs["input_ids"].shape[1]  # número de tokens del prompt
    print(f"Prompt length: {num_tokens} tokens")
 
    # Si quieres avisar si te pasas del contexto:
    if num_tokens + max_new_tokens > tokenizer.model_max_length:
        print(f"El prompt + respuesta ({num_tokens + max_new_tokens}) supera el límite de {tokenizer.model_max_length} tokens.")

    print(f"Max model length: {tokenizer.model_max_length}")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature, 
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
    #exit(0)  # For debugging purposes, remove this line in production
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def call_huggingface_model_pipeline(pipeline, prompt, temperature=0.3, max_new_tokens=300):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Now provide your answer in JSON format as specified."},
    ]

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
    )
    return outputs[0]["generated_text"][-1]["content"]


