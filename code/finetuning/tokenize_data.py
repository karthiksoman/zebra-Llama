from train_utility import *
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
    )
logger = logging.getLogger(__name__)

def tokenize():
    logger.info('Loading the data ...')
    train_data = load_data(config_data['TRAIN_DATA_PATH'])
    logger.info('Data is loaded!')
    
    logger.info('Tokenizing the data ...')
    train_data_tokenized = train_data.map(generate_and_tokenize_prompt)
    logger.info('Data is tokenized!')
    
    logger.info('Saving the tokenized data ...')
    train_data_tokenized.save_to_disk(config_data['TRAIN_DATA_TOKENIZED_PATH'])
    logger.info('Tokenized data is saved!')


if __name__ == '__main__':
    tokenize()
