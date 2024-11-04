from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset, load_from_disk
import torch
import wandb
import os
from config_loader import *

# SYSTEM_PROMPT = '''
# You are an expert in the rare disease Ehlers-Danlos syndrome (EDS).
# You are supposed to answer the question asked by the user.
# Your response should be grounded on the given Context in the user message.
# If no context is given, try to answer as accurately as possible. 
# If you don't know the answer, admit that you don't instead of making one up.   
# '''

SYSTEM_PROMPT = '''
You are an expert AI assistant specializing in Ehlers-Danlos syndrome (EDS). Your role is to provide comprehensive, accurate, and well-structured answers about EDS. You will be provided with a prompt that has two components such as "User message" and "Context". Follow these guidelines to address the prompt:

- In the first paragraph, begin with a broad overview that directly addresses the "User message".
- In the second paragraph, provide detailed information mainly by using the given "Context". Also use your trained knowledge about EDS to supplement the assertions. If you don't see relevant information in the context, always mention that in your response and stick on to your own internal knowledge to answer the question.
- Answer in multiple paragraphs and be comprehensive in your answer
- Structure your response logically:
     a) Start with a general answer to the question.
     b) Provide specific examples or details, always with proper citations. 
     c) You can find the citations at the end of each "Context" para marked as '(Ref: '. Do not use any references that do not contain a DOI, and do not use references that contain just numbers in square brackets. Here are examples of references to avoid: [ 1 ], [5, 6, 8], etc.
- If mentioning specific studies or cases, clearly state their relevance to the main question and provide proper context.
- In the last paragraph, conclude with a brief summary of the key points.
IMPORTANT: If you receive a question unrelated to Ehlers-Danlos Syndrome (EDS), respond directly by stating that the question is not related, without providing any additional context or explanations. For example, if the question is "Who is the actor in the movie titanic" and even if it has any EDS context given in the "Context", your answer should be like "Sorry, this question is not related to EDS and I cannot address that."
'''


def load_data(data_path):
    return load_dataset('json', data_files=data_path, split='train')

def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        config_data['LLM_MODEL_NAME'],
        revision=config_data['LLM_MODEL_BRANCH'],
        cache_dir=config_data['LLM_CACHE_DIR'],
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True,
        rope_scaling={"type": "dynamic", "factor": 8.0}
    )
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

def format_prompt_2(data_sample, tokenizer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": data_sample['question']},
        {"role": "assistant", "content": data_sample['answer']}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def format_prompt_3(data_sample, tokenizer):
    instructions = data_sample["instruction"]
    inputs       = data_sample["input"]
    outputs      = data_sample["output"]
    texts = []
    for instruction_, input_, output_ in zip(instructions, inputs, outputs):        
        user_input = 'Context: ' + input_ + '\n' + 'User message: ' + instruction_
        assistant_response = output_
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_response}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return { "text" : texts, }

def generate_and_tokenize_prompt_3(prompt):
    tokenizer = load_tokenizer()
    return format_prompt_3(prompt, tokenizer)

def format_prompt_for_test_data(data_sample, tokenizer):
    instructions = data_sample["instruction"]
    inputs       = data_sample["input"]
    texts = []
    for instruction_, input_ in zip(instructions, inputs):        
        user_input = 'Context: ' + input_ + '\n' + 'User message: ' + instruction_
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(text)
    return { "text" : texts, }

def generate_and_tokenize_prompt_for_test_data(prompt):
    tokenizer = load_tokenizer()
    return format_prompt_for_test_data(prompt, tokenizer)

def generate_and_tokenize_prompt(prompt):
    tokenizer = load_tokenizer()
    return tokenizer(
        format_prompt_2(prompt, tokenizer),
        truncation=True,
        max_length = config_data['MAX_SEQ_LENGTH'],
        padding='max_length'
    )

def train_model(tokenized_train_dataset, model, tokenizer):
    train_args = TrainingArguments(
        output_dir=os.path.join(config_data['MODEL_OUTDIR'], config_data['CHECKPOINT_SAVE_DIR_NAME']),
        num_train_epochs=config_data['TRAIN_EPOCHS'],
        per_device_train_batch_size=config_data['BATCH_SIZE_PER_GPU_FOR_TRAINING'],
        gradient_accumulation_steps=config_data['GRADIENT_ACCUMULATION_STEPS'],
        optim=config_data['OPTIMIZER'],
        save_strategy=config_data['SAVE_STRATEGY'],
        save_steps=config_data['SAVE_STEPS'],
        logging_steps=config_data['LOGGING_STEPS'],
        learning_rate=float(config_data['LEARNING_RATE']),
        lr_scheduler_type=config_data['LR_SCHEDULER_TYPE'],
        max_steps=config_data['MAX_STEPS'],
        warmup_ratio=config_data['WARMUP_RATIO'],
        max_grad_norm=config_data['MAX_GRAD_NORM'],
        logging_dir=config_data['LOGGING_DIR'],
        report_to='wandb',
        run_name=config_data['WANDDB_RUN_NAME']
    )        
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        args=train_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    trainer.train()

def train_model_2(train_dataset, model, tokenizer):
    train_args = TrainingArguments(
        output_dir=os.path.join(config_data['MODEL_OUTDIR'], config_data['CHECKPOINT_SAVE_DIR_NAME']),
        num_train_epochs=config_data['TRAIN_EPOCHS'],
        per_device_train_batch_size=config_data['BATCH_SIZE_PER_GPU_FOR_TRAINING'],
        gradient_accumulation_steps=config_data['GRADIENT_ACCUMULATION_STEPS'],
        optim=config_data['OPTIMIZER'],
        save_strategy=config_data['SAVE_STRATEGY'],
        save_steps=config_data['SAVE_STEPS'],
        logging_steps=config_data['LOGGING_STEPS'],
        learning_rate=float(config_data['LEARNING_RATE']),
        lr_scheduler_type=config_data['LR_SCHEDULER_TYPE'],
        max_steps=config_data['MAX_STEPS'],
        warmup_ratio=config_data['WARMUP_RATIO'],
        max_grad_norm=config_data['MAX_GRAD_NORM'],
        logging_dir=config_data['LOGGING_DIR'],
        report_to='wandb',
        run_name=config_data['WANDDB_RUN_NAME']
    )        
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        dataset_text_field = "text",
        max_seq_length = config_data['MAX_SEQ_LENGTH'],
        dataset_num_proc = 2,
        packing = False,
        args = train_args
    )
    trainer.train()


