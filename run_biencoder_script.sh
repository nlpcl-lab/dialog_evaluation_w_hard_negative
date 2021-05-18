#!/bin/bash

python train_bi_encoder.py --dataset=persona
python retrieve_with_bi_encoder.py --eval_set=persona --eval_protocal=direct
python retrieve_with_bi_encoder.py --eval_set=persona --eval_protocal=rank
python calculate_similarity.py --eval_set=persona --ir_result_json_fname=./data/cache/persona-response-bi64-epoch0-1000.json
python eval_response_with_retrieved_score.py --eval_set=persona