from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, load_from_disk
import torch
from config_loader import *

SYSTEM_PROMPT = '''
You are an expert in the rare disease Ehlers-Danlos syndrome (EDS).
You are supposed to answer the question asked by the user.
Your response should be grounded on the given Context in the user message.
If no context is given, try to answer as accurately as possible. 
If you don't know the answer, admit that you don't instead of making one up.   
'''

def load_data(data_path):
    return load_dataset('json', data_files=data_path)

def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        config_data['LLM_MODEL_NAME'],
        revision=config_data['LLM_MODEL_BRANCH'],
        cache_dir=config_data['LLM_CACHE_DIR'],
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True)
    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    return model


def apply_lora_config(model):
    lora_config = LoraConfig(
        r=config_data['LORA_r'],
        lora_alpha=config_data['LORA_alpha_fraction']*config_data['LORA_r'],
        lora_dropout=config_data['LORA_dropout'],
        target_modules=config_data['LORA_target_modules'],
        bias='none',
        task_type='CAUSAL_LM'
    )
    return get_peft_model(model, lora_config)
    

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        config_data['LLM_MODEL_NAME'],
        revision=config_data['LLM_MODEL_BRANCH'],
        cache_dir=config_data['LLM_CACHE_DIR'],
        padding_side="left",
        add_eos_token=True,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def format_prompt(data_sample, tokenizer):
    user_input = 'Context: ' + data_sample['input'] + '\n' + 'User message: ' + data_sample['instruction']   
    assistant_response = data_sample['output']
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": assistant_response}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def generate_and_tokenize_prompt(prompt):
    tokenizer = load_tokenizer()
    return tokenizer(
        format_prompt(prompt, tokenizer),
        return_tensors="pt",
        truncation=True,
        max_length = config_data['MAX_SEQ_LENGTH'],
        padding='max_length'
    )
