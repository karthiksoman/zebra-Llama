# zebra-Llama

Zebra-Llama (v0.2) is a specialized version of the Llama-3.1-8b-instruct model, fine-tuned with data specific to the rare disease Ehlers-Danlos Syndrome (EDS) - a rare connective tissue disorder. We utilized textual information from over 4,000 EDS papers from PubMed, more than 8,000 Reddit posts about EDS, and over 5,000 posts from the Inspire forum to gather real-world concerns/questions related to EDS, which were used to fine-tune the model. As a result, this model is adept at providing accurate responses to questions regarding EDS.

The model is trained using a specialized approach called "context-aware training," where we provided context for each question from a custom vector database during the training phase. This approach enabled the model to demonstrate high precision and recall during the inference phase when utilizing the RAG context. Additionally, the model showed a higher likelihood of generating correct citations compared to the base model.
 
# Try zebra-Llama

[Here](https://github.com/karthiksoman/zebra-Llama/tree/main/code/notebook) is the Jupyter Notebook Demo for Zebra-Llama.

[Here](https://zebra-llama-rag.onrender.com/) is the API for the RAG knowledge base that we built for rare diseases, currently focussing on EDS.

# Hugging face Model card

https://huggingface.co/zebraLLAMA/zebra-Llama-v0.2

# Training details

Refer to [config](https://github.com/karthiksoman/zebra-Llama/blob/main/code/finetuning/model_config.yaml) file to know the training parameters

We have also provided the [training](https://github.com/karthiksoman/zebra-Llama/blob/main/code/finetuning/train.py) script that was used to fine-tune the Llama-3.1-8B-Instruct model

<img src="https://github.com/karthiksoman/zebra-Llama/assets/42702311/afacb5ac-1100-47d9-92f5-dbdf3ea0d5b6" style="width: 700px;" />

# Citation

```
@misc{soman2024zebrallamacontextawarelargelanguage,
      title={Zebra-Llama: A Context-Aware Large Language Model for Democratizing Rare Disease Knowledge}, 
      author={Karthik Soman and Andrew Langdon and Catalina Villouta and Chinmay Agrawal and Lashaw Salta and Braian Peetoom and Gianmarco Bellucci and Orion J Buske},
      year={2024},
      eprint={2411.02657},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.02657}, 
}
```

# Team behind zebra-Llama

- [Karthik Soman](https://github.com/karthiksoman)
- [Andrew Langdon](https://github.com/AndrewLngdn)
- [Chinmay Agrawal](https://github.com/ch1nmay7898)
- [Catalina Villouta](https://github.com/mcvillouta)
- [Orion Buske](https://github.com/buske)
- [Lashaw Salta](https://github.com/lashaws)
- [David Harris](https://github.com/d20rvafdln)
