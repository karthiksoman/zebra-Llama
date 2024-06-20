from train_utility import *
from datasets import concatenate_datasets
import random
import logging
import sys

incldue_alpaca = sys.argv[1]
alpaca_fraction = float(sys.argv[2])

wandb.login()
wandb_project = config_data['WANDB_PROJECT_NAME']
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
    )
logger = logging.getLogger(__name__)

def main():
    dataset = load_data(config_data['TRAIN_DATA_PATH'])
    dataset = dataset.map(generate_and_tokenize_prompt_3, batched = True)
    dataset = dataset.train_test_split(test_size=config_data['TEST_DATA_RATIO'], seed=42)    
    
    if incldue_alpaca == 'True':
        alpaca_dataset = load_dataset("tatsu-lab/alpaca", cache_dir=config_data['LLM_CACHE_DIR'])
        total_alpaca_samples = len(alpaca_dataset['train'])
        number_of_alpaca_samples_to_include = round(alpaca_fraction*dataset['train'].num_rows)
        indices = list(range(total_alpaca_samples))
        random.seed(42)
        random.shuffle(indices)
        subset_indices = indices[:number_of_alpaca_samples_to_include]
        alpaca_dataset_subset = alpaca_dataset['train'].select(subset_indices)
        alpaca_dataset_subset = alpaca_dataset_subset.remove_columns('text')
        alpaca_dataset_subset = alpaca_dataset_subset.map(generate_and_tokenize_prompt_3, batched = True)
        train_dataset = concatenate_datasets([dataset['train'], alpaca_dataset_subset])
    else:
        train_dataset = dataset['train']
            
    logger.info('Loading the base model ...')
    model = load_model()
    logger.info('Base model is loaded!')
    
    logger.info('Configuring LoRA ...')
    model = apply_lora_config(model)
    logger.info('LoRA is configured!')
    
    logger.info('Loading the tokenizer ...')
    tokenizer = load_tokenizer()
    logger.info('Tokenizer is loaded!')
    
    logger.info('Start training ...')
    train_model_2(train_dataset, model, tokenizer)
    logger.info('Model is trained and saved!')

if __name__ == '__main__':
    main()




