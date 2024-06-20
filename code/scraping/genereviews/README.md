# Download EDS articles from Genereviews

## Set up python environment (requires Python 3)

```
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

## Download genereviews tarball and extract EDS articles

```
./download.sh
```

## Convert EDS articles from nxml to txt

```
./convert.sh
```

Plaintext version of EDS articles appear in: `gene_reviews_text/`
