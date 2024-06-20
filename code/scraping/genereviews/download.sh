#!/bin/bash -x

# Download genereviews data
if [ ! -e ./gene_NBK1116.tar.gz ]; then
	wget https://ftp.ncbi.nlm.nih.gov/pub/litarch/ca/84/gene_NBK1116.tar.gz
fi
tar -xzf gene_NBK1116.tar.gz

# Keep only nxml articles that relate to EDS
grep -Li "ehlers-danlos" gene_NBK1116/* | xargs -n 5 rm
rm -f gene_NBK1116/*.{png,jpg,pdf,xls,xlsx}
rm -f gene_NBK1116/{TOC,authors,resources_Table3}.nxml
