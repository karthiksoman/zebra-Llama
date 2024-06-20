INPUT_PDF_DIRECTORY_PATH='eds_data/data_sources/research_papers_pdf'
INPUT_XML_DIRECTORY_PATH='eds_data/data_sources/research_papers_xml'
OUTPUT_DIRECTORY_PATH='eds_data/training_data/json_files'
BATCH_SIZE=100
BATCH_DELAY_IN_SEC=50
PAGE_INDEX=0
QUESTIONS_PER_PAGE=2



start_time=$(date +%s)
echo "Running processor.py ..."
python processor.py -p "$INPUT_PDF_DIRECTORY_PATH" -x "$INPUT_XML_DIRECTORY_PATH" -o "$OUTPUT_DIRECTORY_PATH" -b "$BATCH_SIZE" -d "$BATCH_DELAY_IN_SEC" -i "$PAGE_INDEX" -q "$QUESTIONS_PER_PAGE"
wait
end_time=$(date +%s)
time_taken_to_complete=$(( (end_time - start_time) / 60 ))
echo "Completed running processor.py!"
echo "Time taken to complete : $time_taken_to_complete minutes"
