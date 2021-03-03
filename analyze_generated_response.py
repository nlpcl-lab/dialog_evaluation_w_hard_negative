fname = "./attack/neg2_train_k1_maxchange0.5_minchange0.15_nspoveronly0.3.txt"

with open(fname, "r") as f:
    ls = [el.strip().split("|||") for el in f.readlines()]
with open(fname, "r") as f:
    ls2 = [el.strip().split("|||")[-1] for el in f.readlines()]

print(sum([el == "[NONE]" for el in ls2]) / len(ls2))

ls = [[el[-4], el[-3], el[-1]] for el in ls]
for l in ls:
    if l[-1] == "[NONE]":
        continue
    c, r, r_ = l
    print(c)
    print(r)
    print(r_)
    input("")
