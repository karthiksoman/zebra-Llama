from peft import PeftModel
from transformers import TextStreamer
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone import Pinecone
from dotenv import load_dotenv, find_dotenv
import json
import sys
from train_utility import *


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
    

CHECKPOINT = sys.argv[1]
MAX_NEW_TOKENS = 512

## LLM parameters
base_model = load_model()
tokenizer = load_tokenizer()
ft_model = PeftModel.from_pretrained(
    base_model, 
    os.path.join(
        config_data['MODEL_OUTDIR'], 
        config_data['CHECKPOINT_SAVE_DIR_NAME'],
        CHECKPOINT
    )    
)
device = next(ft_model.parameters()).device

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


## LLM + RAG
user_message_ = input('User message: ')
input_ = get_rag_context(user_message_)
user_input = 'Context: ' + input_ + '\n' + 'User message: ' + user_message_ + '\n' + "Always make sure to provide references in your answer. You can find the references in the Context marked as '(Ref: '. After providing references, print 'End of response' "
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_input}
]
input_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_input = tokenizer(input_prompt, return_tensors="pt").to(device)
ft_model.eval()

with torch.no_grad():
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    output = ft_model.generate(**model_input, 
                               streamer=streamer,
                               eos_token_id=ft_model.generation_config.eos_token_id,
                               stop_strings=["End of response"],
                               tokenizer=tokenizer,
                               max_new_tokens=MAX_NEW_TOKENS,
                               temperature=config_data['LLM_TEMPERATURE'],
                               do_sample=True)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)


