fname = "./data/negative/neg2_train_dialoggpt-prefix-topk50.txt"

with open(fname, "r") as f:
    ls = [el.strip().split("|||") for el in f.readlines()]

org, pr = [], []
ratio = []
for l in ls:
    if l[3] == "[NONE]":
        ratio.append(0)
        continue
    ratio.append(1)
    org.append(len(l[1].split()))
    pr.append(len(l[3].split()))
    # print(l[1])
    # print(l[3])
    # print("\n\n")
    # input()

print(sum(org) / len(org))
print(sum(pr) / len(pr))
print(sum(ratio) / len(ratio))
