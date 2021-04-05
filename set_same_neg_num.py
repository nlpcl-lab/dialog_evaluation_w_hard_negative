import os


ours_fname = "./attack/neg2_{}_k5_maxchange0.4_minchange0.1_NSPCUT0.4.txt"
rand2_fname = "./data/negative/random_neg2_{}.txt"
output_fname = "./data/negative/random_negsame_{}.txt"

drop_ratio = 0.2  # 두번째 randneg 중 20퍼센트 날림
import random

for setname in ["train", "valid"]:
    with open(rand2_fname.format(setname), "r") as f:
        ls = [el.strip().split("|||") for el in f.readlines()]
    sampled_idx = random.sample(range(0, len(ls)), int(0.2 * len(ls)))
    output_data = []
    for idx, line in enumerate(ls):
        final_sample = "[NONE]" if idx in sampled_idx else line[-1]
        assert len(line) == 4
        line[-1] = final_sample
        output_data.append("|||".join(line))
    with open(output_fname.format(setname), "w") as f:
        f.write("\n".join(output_data))
