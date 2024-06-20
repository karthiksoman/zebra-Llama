from peft import PeftModel
from transformers import TextStreamer
from train_utility import *

CHECKPOINT = 'checkpoint-7500'
MAX_NEW_TOKENS=512


base_model = AutoModelForCausalLM.from_pretrained(
    config_data['LLM_MODEL_NAME'],
    cache_dir=config_data['LLM_CACHE_DIR'],
    device_map='auto',
    trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(
    config_data['LLM_MODEL_NAME'],
    cache_dir=config_data['LLM_CACHE_DIR'],
    padding_side="left",
    add_eos_token=True,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

ft_model = PeftModel.from_pretrained(
    base_model, 
    os.path.join(config_data['MODEL_OUTDIR'], 
                 config_data['CHECKPOINT_SAVE_DIR_NAME'],
                 CHECKPOINT)
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
'''

def formatting_input_prompts_func(instruction_, input_):
    text = eds_prompt.format(instruction_, input_)
    return text

user_message_ = input('User message: ')
input_ = ''
eval_prompt = formatting_input_prompts_func(user_message_, input_)

device = next(ft_model.parameters()).device
model_input = tokenizer(eval_prompt, return_tensors="pt").to(device)
ft_model.eval()

with torch.no_grad():
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    output = ft_model.generate(**model_input, 
                               streamer=streamer, 
                               max_new_tokens=MAX_NEW_TOKENS,
                               temperature=config_data['LLM_TEMPERATURE'],
                               do_sample=True)
    output_text = tokenizer.decode(output[0], 
                                   skip_special_tokens=True)

