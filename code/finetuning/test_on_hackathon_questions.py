from peft import PeftModel
from transformers import TextStreamer
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone import Pinecone
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import json
import sys
from tqdm import tqdm
from train_utility import *


USE_FINETUNE_MODEL = sys.argv[1]
CHECKPOINT = None
MAX_NEW_TOKENS = 512

base_model = load_model()
tokenizer = load_tokenizer()

if USE_FINETUNE_MODEL == 'True':
    CHECKPOINT = sys.argv[2]
    ft_model = PeftModel.from_pretrained(
        base_model, 
        os.path.join(
            config_data['MODEL_OUTDIR'], 
            config_data['CHECKPOINT_SAVE_DIR_NAME'],
            CHECKPOINT
        )    
    )
    ## RAG parameters
    load_dotenv(os.path.join(os.path.expanduser('~'), '.openai_config.env'))
    load_dotenv(os.path.join(os.path.expanduser('~'), '.pinecone_config.env'))
    embed_model = OpenAIEmbedding(
        model=config_data['RAG_EMBEDDING_MODEL'],
        api_key=os.environ.get('HACKATHON_API_KEY'),
    )
    Settings.embed_model = embed_model
    pc = Pinecone(api_key=os.environ.get('ANDREW_API_KEY'))
    pinecone_index = pc.Index(config_data['RAG_PINECONE_INDEX'])
    model = ft_model
    save_filename = 'hackathon_test_response_from_zebraLLAMA.csv'
else:
    model = base_model    
    save_filename = 'hackathon_test_response_from_LLAMA.csv'
device = next(model.parameters()).device


def main():
    test_data = load_data(config_data['HACKATHON_TEST_DATA'])
    if USE_FINETUNE_MODEL == 'True':
        test_data = test_data.map(update_input_with_rag_context)
        test_data = test_data.map(append_to_instruction)        
    else:
        test_data = test_data.map(update_input_with_empty_context)
        
    test_data = test_data.map(generate_and_tokenize_prompt_for_test_data, batched = True)
    eval_prompts = list(map(lambda x:x['text'], test_data))
    eval_prompts_batches = batchify(eval_prompts, config_data['TEST_BATCH_SIZE'])
    
    test_output = []
    for eval_prompts_batch in tqdm(eval_prompts_batches):
        test_output.extend(get_batch_response(eval_prompts_batch, model, tokenizer))
    test_output_df = pd.DataFrame(test_output, columns=['prompt', 'response'])
    test_output_df.to_csv(f'../../eds_data/{save_filename}', index=False, header=True)


def append_to_instruction(example):
    append_text = "\nAlways make sure to provide references in your answer. You can find the references in the Context marked as '(Ref: '. After providing references, print 'End of response'"
    example['instruction'] = example['instruction'] + append_text
    return example

def batchify(mylist, batch_size):
    batches = []
    for i in range(0, len(mylist), batch_size):  
        batches.append(mylist[i:i+batch_size])
    return batches

def get_rag_context(query, top_k=5):
    query_embedding = embed_model.get_text_embedding(
        query
    )
    retrieved_doc = pinecone_index.query(
        vector=query_embedding, 
        top_k=top_k, 
        include_metadata=True
    )
    extracted_context_summary = list(map(lambda x:json.loads(x.metadata['_node_content'])['metadata']['section_summary'], retrieved_doc.matches))
    provenance = list(map(lambda x:x.metadata['c_document_id'], retrieved_doc.matches))
    context = ''
    for i in range(top_k):
        context += extracted_context_summary[i] + '(Ref: ' + provenance[i] + '). '
    return context

def update_input_with_rag_context(example):
    query = example['instruction']
    new_input = get_rag_context(query)
    example['input'] = new_input
    return example

def update_input_with_empty_context(example):
    example['input'] = ''
    return example

def get_batch_response(eval_prompts_batch, model, tokenizer):
    model_input = tokenizer(eval_prompts_batch, 
                            padding=True, 
                            return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(**model_input, 
                                       eos_token_id=model.generation_config.eos_token_id,
                                       stop_strings=["End of response"],
                                       tokenizer=tokenizer,
                                       max_new_tokens=MAX_NEW_TOKENS,
                                       temperature=config_data['LLM_TEMPERATURE'],
                                       do_sample=True)
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    user_input = list(map(lambda x:x.split('\nUser message: ')[-1].split('\nAlways make sure to provide references in your answer.')[0].strip(), generated_texts))
    llm_response = list(map(lambda x:x.split('assistant\n\n')[-1].split(' End of response')[0].strip(), generated_texts))
    return list(zip(user_input, llm_response))



if __name__ == "__main__":
    main()


    








# def get_response(user_message_, model, tokenizer, checkpoint=None):
#     if not checkpoint:
#         input_ = get_rag_context(user_message_)
#         user_input = 'Context: ' + input_ + '\n' + 'User message: ' + user_message_ + '\n' + "Always make sure to provide references in your answer. You can find the references in the Context marked as '(Ref: '. After providing references, print 'End of response' "
#     else:
#         input_ = ' '
#         user_input = 'Context: ' + input_ + '\n' + 'User message: ' + user_message_
    
#     messages = [
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user", "content": user_input}
#     ]
#     input_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
#     model_input = tokenizer(input_prompt, return_tensors="pt").to(device)
#     model.eval()    
#     with torch.no_grad():
#         output = model.generate(**model_input, 
#                                    eos_token_id=model.generation_config.eos_token_id,
#                                    stop_strings=["End of response"],
#                                    tokenizer=tokenizer,
#                                    max_new_tokens=MAX_NEW_TOKENS,
#                                    temperature=config_data['LLM_TEMPERATURE'],
#                                    do_sample=True)
#         output_text = tokenizer.decode(output[0], skip_special_tokens=True)
#     return output_text