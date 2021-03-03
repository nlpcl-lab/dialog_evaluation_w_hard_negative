import os


original_fname = './data/negative/neg2_{}_k5_maxchange0.5_minchange0.15_nspoveronly0.3.txt'

for setname in ['valid','train']:
    fname = original_fname.format(setname)
    changed_fname = fname.replace('neg2','neg1')
    assert os.path.exists(fname) and not os.path.exists(changed_fname)
    
    with open(fname,'r') as f:
        ls = [el.strip().split("|||") for el in f.readlines()]
        assert all([len(el)==4 for el in ls])
    new_ls = []
    for line in ls:
        c,g,r,n = line
        new_ls.append("|||".join([c,g,n]))
    with open(changed_fname,'w') as f:
        f.write("\n".join(new_ls))
    