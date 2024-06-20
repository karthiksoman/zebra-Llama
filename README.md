# zebra-Llama

Zebra-Llama is a specialized version of the Llama-3-8b-instruct model, fine-tuned using data specific to Ehlers-Danlos syndrome (EDS). We utilized textual information from over 4,000 EDS papers from PubMed, more than 8,000 Reddit EDS posts (publicly available), and over 5,000 EDS posts from the Inspire forum (publicly available) to refine the model. In addition to this, we also indexed Pinecone vectorDB with more than 50,000 EDS related information. As a result, this model is adept at providing accurate responses to questions related to EDS. 

# Try zebra-Llama

We have built a UI for zebra-Llama where users can try out questions related to Ehlers-Danlos syndrome (EDS).

https://zebra-llama-ui.streamlit.app/

# Hugging face Model card

https://huggingface.co/zebraLLAMA/zebra-Llama-v0.1

# Training details

Refer to [config](https://github.com/karthiksoman/zebra-Llama/blob/main/code/finetuning/model_config.yaml) file to know the training parameters

We have also provided the [training](https://github.com/karthiksoman/zebra-Llama/blob/main/code/finetuning/train.py) script that was used to fine-tune the Llama-3-8b-instruct model

<img src="https://github.com/karthiksoman/zebra-Llama/assets/42702311/afacb5ac-1100-47d9-92f5-dbdf3ea0d5b6" style="width: 700px;" />

# Team behind zebra-Llama

- [Karthik Soman](https://github.com/karthiksoman)
- [Andrew Langdon](https://github.com/AndrewLngdn)
- [Chinmay Agrawal](https://github.com/ch1nmay7898)
- [Catalina Villouta](https://github.com/mcvillouta)
- [Orion Buske](https://github.com/buske)
- [Lashaw Salta](https://github.com/lashaws)
- [David Harris](https://github.com/d20rvafdln)
