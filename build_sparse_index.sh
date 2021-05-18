#!/bin/bash
python run_sparse_retrieval.py --eval_set=dd
python run_sparse_retrieval.py --eval_set=persona

python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator \
 -threads 1 -input dd-docs \
 -index indexes/dd_index.jsonl -storePositions -storeDocvectors -storeRaw

python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator \
 -threads 1 -input persona-docs \
 -index indexes/persona_index.jsonl -storePositions -storeDocvectors -storeRaw



