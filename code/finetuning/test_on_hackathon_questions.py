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
import torch
from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
import re
from collections import Counter
from threading import Thread
import joblib
import requests

system_prompt = '''
You are an expert AI assistant specializing in Ehlers-Danlos syndrome (EDS). Your role is to provide comprehensive, accurate, and well-structured answers about EDS. You will be provided with a prompt that has two components such as "User message" and "Context". Follow these guidelines to address the prompt:

- In the first paragraph, begin with a broad overview that directly addresses the "User message".
- In the second paragraph, provide detailed information mainly by using the given "Context". Also use your trained knowledge about EDS to supplement the assertions. If you don't see relevant information in the context, always mention that in your response and stick on to your own internal knowledge to answer the question.
- Answer in multiple paragraphs and be comprehensive in your answer
- Structure your response logically:
     a) Start with a general answer to the question.
     b) Provide specific examples or details, always with proper citations. 
     c) You can find the citations at the end of each "Context" para marked as '(Ref: '. Do not use any references that do not contain a DOI, and do not use references that contain just numbers in square brackets. Here are examples of references to avoid: [ 1 ], [5, 6, 8], etc.
- If mentioning specific studies or cases, clearly state their relevance to the main question and provide proper context. 
- When answering questions based on provided context, do not use phrases like 'The context provided' or 'In the provided context' in your responses.
- In the last paragraph, conclude with a brief summary of the key points.
IMPORTANT: If you receive a question unrelated to Ehlers-Danlos Syndrome (EDS), respond directly by stating that the question is not related, without providing any additional context or explanations. For example, if the question is "Who is the actor in the movie titanic" and even if it has any EDS context given in the "Context", your answer should be like "Sorry, this question is not related to EDS and I cannot address that."

Some examples for your reference are given below:
Example 1:
Context : Context not related to EDS
User message: What is the most popular song of Taylor Swift?
Your response : Sorry, this question is not directly related to EDS. So, I cannot answer that.

Example 2:
Context : The analysis of the hEDS patients’ ﬁbroblasts showed that there is a greater expression
of AQP9 gene and down regulation of IL6 gene. AQP9 and IL6 are involved in the
inﬂammatory process, since EDS patients frequently show chronic disease, AQP9 and IL6
could be use as possible diagnostic biomarkers.
In addition, PRL may have a role in inﬂammation and their serum levels could suggest
a clinical situation of chronic inﬂammation that is frequent in EDS patients. Scientiﬁc data
show the up-regulation of SELENBP1 in schizophrenic patients. Considering that EDS
patients suffer from psychiatric disorders, it would be interesting to dose SELENBP1 in
serum EDS patients.
User message: What are the biomarkers associated with EDS?
Your response: Biomarkers are measurable indicators that can be used to assess the presence or severity of a disease. For Ehlers-Danlos Syndrome (EDS), research has identified several potential biomarkers that could help in diagnosing or understanding the disease better. EDS is a group of connective tissue disorders characterized by hypermobility, skin elasticity, and tissue fragility. Identifying reliable biomarkers is crucial for effective diagnosis and management, given the complex and varied symptoms associated with different types of EDS.
One of the potential biomarkers in EDS is the **Aquaporin 9 (AQP9) gene**, which has been shown to be overexpressed in the fibroblasts of hypermobile EDS (hEDS) patients. AQP9 plays a role in the inflammatory process, and its increased expression may be linked to the chronic inflammation often observed in EDS patients. Additionally, the **Interleukin-6 (IL6) gene**, which is also involved in inflammation, has been found to be downregulated in these patients. This dysregulation of genes related to the inflammatory response suggests that both AQP9 and IL6 could serve as useful diagnostic markers in identifying inflammatory states associated with EDS.
Furthermore, **Prolactin (PRL)** has been suggested as another potential biomarker. Elevated serum levels of PRL may indicate a state of chronic inflammation, which is frequently observed in EDS patients. Another interesting biomarker is **SELENBP1** (selenium-binding protein 1), which is known to be upregulated in patients with schizophrenia. Considering the higher prevalence of psychiatric disorders in EDS patients, the measurement of SELENBP1 levels could be relevant in the context of EDS-related psychiatric manifestations. These findings, while still in the early stages of research, could pave the way for better diagnostic tools and therapeutic targets for EDS (Ref: 10.3390/ijms221810149).
In summary, biomarkers such as AQP9, IL6, PRL, and SELENBP1 are being explored in the context of EDS to help clarify the underlying mechanisms of the disease and potentially improve diagnostic accuracy. Further research and validation are necessary to establish these markers as definitive diagnostic tools.
'''


SAVE_RESULT_FILENAME = "hackathon_test_data_out.joblib"

## RAG parameters
RAG_BASE_URI = "https://zebra-llama-rag.onrender.com"
RAG_ENDPOINT = "/search"
RAG_URI = RAG_BASE_URI + RAG_ENDPOINT
TOP_K = 2

## Set embedding model
load_dotenv(os.path.join(os.path.expanduser('~'), '.openai_config.env'))
embed_model = OpenAIEmbedding(
    model=config_data['RAG_EMBEDDING_MODEL'],
    api_key=os.environ.get('HACKATHON_API_KEY'),
)
Settings.embed_model = embed_model


## Load finetuned and basemodel
ft_model = AutoModelForCausalLM.from_pretrained("zebraLLAMA/zebra-Llama-v0.2")
base_model = load_model()    
tokenizer = load_tokenizer()

device = next(ft_model.parameters()).device


def main():
    repetition_stopping = RepetitionStoppingCriteria(tokenizer, repetition_threshold=3, window_size=200, min_length=20)
    stopping_criteria = StoppingCriteriaList([repetition_stopping])
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

    test_data = load_data(config_data['HACKATHON_TEST_DATA'])
    
    finetuned_model_generated_text_out = []
    count = 0
    for item in tqdm(test_data):
        query = item["instruction"]
        resp = generate_text_using_stream(query, ft_model)
        if len(resp.split("\n")) > 1:
            finetuned_model_generated_text_out.append((query, resp))

    base_model_generated_text_out = []
    count = 0
    for item in tqdm(test_data):
        query = item["instruction"]
        resp = generate_text_using_stream(query, base_model)
        if len(resp.split("\n")) > 1:
            base_model_generated_text_out.append((query, resp))

    test_data_out = {
        "finetuned_model" : finetuned_model_generated_text_out,
        "base_model" : base_model_generated_text_out
    }
    joblib.dump(test_data_out, SAVE_RESULT_FILENAME)
    
    

def get_rag_context(query, rag_uri, top_k=2):
    query_embedding = embed_model.get_text_embedding(query)
    response = requests.post(
        rag_uri,
        json={
            "query_embedding": query_embedding,
            "top_k": top_k
        }
    )
    if response.status_code == 200:
        return response.json()["context"]
    else:
        return ''

class RepetitionStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, repetition_threshold=3, window_size=200, min_length=20):
        self.tokenizer = tokenizer
        self.repetition_threshold = repetition_threshold
        self.window_size = window_size
        self.min_length = min_length
        self.generated_text = ""
        self.last_check_length = 0

    def __call__(self, input_ids, scores, **kwargs):
        new_text = self.tokenizer.decode(input_ids[0, self.last_check_length:], skip_special_tokens=True)
        self.generated_text += new_text
        self.last_check_length = len(input_ids[0])

        if len(self.generated_text) > self.window_size and self.check_repetition():
            return True

        return False

    def check_repetition(self):
        text = self.generated_text[-self.window_size:]
        sentences = re.split(r'[.!?]+', text)
        
        # Check for exact sentence repetitions
        sentence_counter = Counter(sentences)
        if any(count >= self.repetition_threshold for count in sentence_counter.values()):
            return True

        # Check for phrase repetitions
        phrases = self.get_phrases(text)
        phrase_counter = Counter(phrases)
        if any(count >= self.repetition_threshold and len(phrase) >= self.min_length for phrase, count in phrase_counter.items()):
            return True

        return False

    def get_phrases(self, text):
        words = text.split()
        phrases = []
        for i in range(len(words)):
            for j in range(i+1, len(words)+1):
                phrase = ' '.join(words[i:j])
                if len(phrase) >= self.min_length:
                    phrases.append(phrase)
        return phrases

    
def generate_text_using_stream(query, model, return_out=True):
    
    rag_context = get_rag_context(query, RAG_URI, top_k=TOP_K)
    prompt = f'''
    Context: {rag_context}
    User message: {query}
    '''
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_input = tokenizer(text, 
                            padding=True, 
                            return_tensors="pt").to(device)
    
    generation_kwargs = dict(
        **model_input,
        max_new_tokens=1024,
        top_p=0.95,
        top_k=50,
        do_sample=True,
        streamer=streamer,
        stopping_criteria=StoppingCriteriaList([stopping_criteria])
    )
    
    # Start the generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Iterate over the generated text
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        print(new_text, end="", flush=True)  # Print each piece of new text as it's generated
    
    thread.join()
    clear_output()
    if return_out:
        return generated_text


if __name__ == "__main__":
    main()