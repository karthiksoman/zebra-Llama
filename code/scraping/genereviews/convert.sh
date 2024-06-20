#!/bin/bash

# Convert genereviews nxml articles to plaintext for vector database
mkdir gene_reviews_text
for f in gene_NBK1116/*.nxml; do
	outfile=gene_reviews_text/$(basename $f .nxml).txt
	echo "Processing ${f}"
	python3 clean_gene_reviews.py $f > $outfile
	echo "  => ${outfile}"
done
