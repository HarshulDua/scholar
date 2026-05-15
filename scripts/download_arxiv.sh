#!/usr/bin/env bash
# Downloads the ArXiv metadata snapshot from Kaggle
# Requires: pip install kaggle && kaggle API key configured
kaggle datasets download -d Cornell-University/arxiv -p ./data --unzip
