from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import torch
from train_utility import *

dtype = None
load_in_4bit = True


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = config_data['LLM_MODEL_NAME'],
    max_seq_length = config_data['MAX_SEQ_LENGTH'],
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    cache_dir=config_data['LLM_CACHE_DIR'],
    device_map='auto'
)
if torch.cuda.device_count() > 1:
    model.is_parallelizable = True
    model.model_parallel = True
        
model = FastLanguageModel.get_peft_model(
    model,
    r = config_data['LORA_r'],
    target_modules = config_data['LORA_target_modules'],
    lora_alpha = config_data['LORA_alpha_fraction']*config_data['LORA_r'],
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = True, 
    random_state = 3407,
    use_rslora = False,  
    loftq_config = None
)


eds_prompt = '''
You are an expert in the rare disease called Ehlers-Danlos syndrome (EDS).
Below is an instruction that describes a question asked by the user, paired with an input that provides further context. Write a response that appropriately completes the request.
If any references is given in the context (as 'Ref'), make sure to include that citation in your response.
If no context is given, try to answer as accurately as possible. 
If you don't know the answer, admit that you don't instead of making one up.   

### Instruction:
{}

### Input:
{}

### Response:
{}
'''

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction_, input_, output_ in zip(instructions, inputs, outputs):
        text = eds_prompt.format(instruction_, input_, output_) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }





if not os.path.exists(config_data['TRAIN_DATA_TOKENIZED_PATH']):
    dataset = load_data(config_data['TRAIN_DATA_PATH'])
    dataset = dataset.map(formatting_prompts_func, batched = True)
    dataset.save_to_disk(config_data['TRAIN_DATA_TOKENIZED_PATH'])
else:
    dataset = load_from_disk(config_data['TRAIN_DATA_TOKENIZED_PATH'])

dataset = dataset.train_test_split(test_size=config_data['TEST_DATA_RATIO'], seed=42)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset['train'],
    dataset_text_field = "text",
    max_seq_length = config_data['MAX_SEQ_LENGTH'],
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = config_data['GRADIENT_ACCUMULATION_STEPS'],
        warmup_ratio=config_data['WARMUP_RATIO'],
        max_steps = config_data['MAX_STEPS'],
        learning_rate = float(config_data['LEARNING_RATE']),
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = config_data['LOGGING_STEPS'],
        logging_dir=config_data['LOGGING_DIR'],
        optim = config_data['OPTIMIZER'],
        lr_scheduler_type = config_data['LR_SCHEDULER_TYPE'],        
        output_dir = os.path.join(config_data['MODEL_OUTDIR'], config_data['CHECKPOINT_SAVE_DIR_NAME']),
        num_train_epochs=config_data['TRAIN_EPOCHS'],
        max_grad_norm=config_data['MAX_GRAD_NORM'],        
        save_strategy=config_data['SAVE_STRATEGY'],
        save_steps=config_data['SAVE_STEPS'],
        report_to='wandb',
        run_name=config_data['WANDDB_RUN_NAME'],
        seed = 3407,
    ),
)

trainer_stats = trainer.train()

