from utility import (
    REPO_ROOT_PATH,
    ParseException,
    create_question_from_file_v2,
    create_question_from_text_v2,
)
from llm import get_llm
import random
import json
from pathlib import Path
from tqdm import tqdm as print_progress
import time
import concurrent.futures
import xml.etree.ElementTree as ET
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, default="eds_data/data_sources/research_papers_pdf",  help='Provide name of the input directory where pdf files to process are saved')
parser.add_argument('-x', type=str, default="eds_data/data_sources/research_papers_xml",  help='Provide name of the input directory where xml files to process are saved')
parser.add_argument('-o', type=str, default="eds_data/training_data/json_files", help='Provide name of the output directory to save the output JSON file')
parser.add_argument('-b', type=int, default=100, help='Provide batch size')
parser.add_argument('-d', type=int, default=50, help='Provide batch delay in sec')
parser.add_argument('-i', type=int, default=0, help='if format is pdf, which page number to consider for creating questions')
parser.add_argument('-q', type=int, default=2, help='if format is pdf, how many questions per page should be created')

args = parser.parse_args()


DEFAULT_INPUT_PATH = (
    REPO_ROOT_PATH / "eds_data" / "data_sources" / "research_papers_pdf"
)
DEFAULT_OUTPUT_PATH = REPO_ROOT_PATH / "eds_data" / "training_data" / "json_files"


class DirectoryProcessor:
    def __init__(
        self,
        input_directory_path=None,
        output_directory_path=None,
        input_extension=".pdf",
        output_extension=".json",
        batch_size=20,
        batch_delay_sec=None,
    ):
        """
        This processor will look at the files in 'input_directory_path` and check which are missing from
        'output_directory_path' and process them.

        input_directory_path: relative from project root.  Default: eds_data/data_sources/research_papers_pdf
        """
        if input_directory_path:
            self.input_directory_path = REPO_ROOT_PATH / input_directory_path
        else:
            self.input_directory_path = DEFAULT_INPUT_PATH

        if output_directory_path:
            self.output_directory_path = REPO_ROOT_PATH / output_directory_path
        else:
            self.output_directory_path = DEFAULT_OUTPUT_PATH

        self.batch_size = batch_size
        self.input_extension = input_extension
        self.output_extension = output_extension
        self.batch_delay_sec = batch_delay_sec

    def run(self):
        input_file_paths = list(
            self.input_directory_path.glob(f"*{self.input_extension}")
        )
        input_file_names = [p.name for p in input_file_paths]
        print(
            f"Found {len(input_file_names)} total {self.input_extension} files in {self.input_directory_path}."
        )

        output_file_paths = list(
            self.output_directory_path.glob(f"*{self.output_extension}")
        )
        output_file_names = [p.name for p in output_file_paths]
        print(
            f"Found {len(output_file_names)} {self.output_extension} files in {self.output_directory_path}."
        )

        processed_file_names = [
            name.replace(self.output_extension, self.input_extension)
            for name in output_file_names
        ]

        unprocessed_file_names = set(input_file_names).difference(
            set(processed_file_names)
        )
        print(
            f"Found {len(unprocessed_file_names)} unprocessed {self.input_extension} files in {self.input_directory_path}."
        )

        unprocessed_file_paths = [
            self.input_directory_path / name for name in unprocessed_file_names
        ]
        unprocessed_file_paths = [str(p.absolute()) for p in unprocessed_file_paths]

        random.shuffle(unprocessed_file_paths)

        for i in print_progress(range(0, len(unprocessed_file_paths), self.batch_size)):
            path_batch = unprocessed_file_paths[i : i + self.batch_size]
            self._process_batch(path_batch)

            if self.batch_delay_sec:
                print(
                    f"Sleeping {self.batch_delay_sec} seconds to avoid API rate limits..."
                )
                time.sleep(self.batch_delay_sec)

    def _process_batch(self, path_batch):
        print("Starting batch...")

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(path_batch)
        ) as executor:
            futures = set(
                executor.submit(self._process_one, file_path)
                for file_path in path_batch
            )

    def _process_one(self, file_path):
        """
        Override in subclass
        """

    def _write_data(self, output, file_path):
        file_name = Path(
            file_path.replace(self.input_extension, self.output_extension)
        ).name
        output_path = self.output_directory_path / file_name

        with open(output_path, "w") as json_file:
            json.dump(output, json_file, indent=4, sort_keys=True)

        print(f"Wrote file {output_path}")


class PDFDirectoryProcessor(DirectoryProcessor):
    def _process_one(self, file_path):
        try:
            training_data = create_question_from_file_v2(
                file_path, page_index=args.i, number_of_questions_per_page=args.q
            )

            if isinstance(training_data, dict):
                training_data = [training_data]

            if not isinstance(training_data, list):
                print(f"training_data has an unknown format. Contents: {training_data}")
                return

            self._write_data(training_data, file_path)

        except ParseException as e:
            print(e)
            return

        except Exception as e:
            print(f"Exception when processing {file_path}:")
            print(e)
            return


class XMLDirectoryProcessor(DirectoryProcessor):
    def _process_one(self, file_path):
        try:
            tree = ET.parse(file_path)
            abstract_node = tree.find(".//abstract")
            abstract_texts = [t for t in abstract_node.itertext()]
            abstract = " ".join(abstract_texts)

            training_data = create_question_from_text_v2(
                abstract, number_of_questions=2
            )
            if isinstance(training_data, dict):
                training_data = [training_data]

            if not isinstance(training_data, list):
                print(f"training_data has an unknown format. Contents: {training_data}")
                return

            self._write_data(training_data, file_path)

        except ParseException as e:
            print(e)
            return

        except Exception as e:
            print(f"Exception when processing {file_path}:")
            print(e)
            return

def prune_files():
    files = os.listdir(os.path.join(REPO_ROOT_PATH, args.o))
    files_corrupted = []
    for file in files:	
    	with open(os.path.join(REPO_ROOT_PATH, args.o, file), 'r') as f:
    		try:
    			data = json.load(f)
    		except:
    			files_corrupted.append(file)
    if len(files_corrupted) > 0: 
        print(f'{len(files_corrupted)} files have corrupted JSON entries. Hence, removing them from the output directory ...')
        for file_corrupted in files_corrupted:
        	os.remove(os.path.join(REPO_ROOT_PATH, args.o, file_corrupted))
    else:
        print('There are no corrupted files!')
    print('Pruning completed!')

def main():
    XMLDirectoryProcessor(
        batch_size=args.b,
        batch_delay_sec=args.d,
        input_extension=".xml",
        input_directory_path=args.x,
        output_directory_path=args.o,
    ).run()

    PDFDirectoryProcessor(
        batch_size=args.b,
        batch_delay_sec=args.d,
        input_directory_path=args.p,
        output_directory_path=args.o,
    ).run()
    prune_files()

if __name__ == "__main__":
    main()


