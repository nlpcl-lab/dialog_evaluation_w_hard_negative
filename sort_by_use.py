from sklearn.metrics.pairwise import cosine_similarity
import os
from utils import get_usencoder

import random
from tqdm import tqdm

USE_USE_RANK = False
top_n = 10
USE_DIALOG_GPT = True

input_template = './data/negative/prefix-beam{}_neg2_{}.txt'
if USE_USE_RANK:
    use_model = get_usencoder()
    output_template = './data/negative/neg2_{}_prefix-beam{}_use_rank.txt'
else:
    output_template = './data/negative/neg2_{}_prefix-beam{}.txt'

if USE_DIALOG_GPT:
    input_template = input_template.replace("prefix-", 'dialoggpt-prefix-')
    output_template = output_template.replace('prefix-', 'dialoggpt-prefix-')


def read_file(fname, num_in_line=3):

    final_output = []
    with open(fname, 'r') as f:
        ls = [el.strip().split("|||") for el in f.readlines()]
    for idx, line in enumerate(ls):
        if len(line) == num_in_line:
            final_output.append(line)

    assert len(final_output) != 0
    return final_output


for setname in ['valid', 'train']:
    num_in_line = 4
    input_fname = input_template.format(top_n, setname)
    output_fname = output_template.format(setname, top_n)
    print(input_fname, end=' -> ')
    print(output_fname)
    input("Right?\n")
    assert not os.path.exists(output_fname)

    input_data = read_file(input_fname, num_in_line=num_in_line)
    output_data = []
    for sample_idx, sample in enumerate(tqdm(input_data)):
        context, response, random_neg, generated = sample[0], sample[1], sample[2], sample[3:]
        assert len(generated) == num_in_line-3
        assert len(list(set([el[:3] for el in [response]+generated])))
        if USE_USE_RANK:
            golden_emb = use_model([response])[0]
            generated_emb = use_model(generated)
            cossim = cosine_similarity([golden_emb], generated_emb)[0]
            most_dissimilar_generated = generated[cossim.argmin()]
            output_data.append(
                [context, response, random_neg, most_dissimilar_generated])
        else:
            generated = random.sample(generated, 1)[0]
            assert isinstance(generated, str)
            output_data.append(
                [context, response, random_neg, generated])

    with open(output_fname, 'w') as f:
        for line_idx, line in enumerate(output_data):
            assert len(line) == 4
            f.write("|||".join(line))
            if line_idx != len(output_data)-1:
                f.write('\n')
