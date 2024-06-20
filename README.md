# zebra-Llama

Zebra-Llama is a specialized version of the Llama-3-8b-instruct model, fine-tuned using data specific to EDS. We utilized textual information from over 4,000 EDS papers from PubMed, more than 8,000 Reddit EDS posts (publicly available), and over 5,000 EDS posts from the Inspire forum (publicly available) to refine the model. In addition to this, we also indexed Pinecone vectorDB with more than 50,000 EDS related information. As a result, this model is adept at providing accurate responses to questions related to EDS. 


# Hugging face Model card

https://huggingface.co/zebraLLAMA/zebra-Llama-v0.1

# Training details

Refer to [config](https://github.com/karthiksoman/zebra-Llama/blob/main/code/finetuning/model_config.yaml) file to know the training parameters

We have also provided the [training](https://github.com/karthiksoman/zebra-Llama/blob/main/code/finetuning/train.py) script that was used to fine-tune the Llama-3-8b-instruct model
