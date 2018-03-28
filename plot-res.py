import sys
import json
import re

import numpy as np
import matplotlib.pyplot as plt

def tag_info(tag):
    i = { 'aeg':   (50, "AEG")
        , 'ucb1':  (60, "UCB 1")
        , 'bayes': (10,  "Bayes 1")
        , 'rnd':   (20,  "Bayes 2")
        , 'blr':   (30, "Gr Exp 1")
        , 'bls':   (40, "Gr Exp 2")
    }

    order, name = i[tag]
    h = { 'name': name, 'order': order }
    return h

def tag_name(tag):
    ti = tag_info(tag)
    return ti['name']

def tag_order(tag):
    ti = tag_info(tag)
    return ti['order']

def tidy_keys(d):

    kss = list(d.keys())

    # calculate how many different values are seen for each bit of the key...
    k = [ {} for k in list(kss[0]) ]
    
    for ks in kss:
        i = 0
        for dk in list(ks):
            k[i][dk] = 1
            i += 1

    # .. and store in nks
    nks = [ len(kd.keys()) for kd in k ]

    tidy_d = {}

    for k,v in d.items():
        tidy_k = [ str(k) for (k,n) in zip(k,nks) if n > 1]
        if len(tidy_k) == 1:
            tidy_k = tidy_k[0]
        else:
            tidy_k = ', '.join(tidy_k)

        tidy_d[tidy_k] = v

    common_keys = ', '.join([ k for (k,n) in zip(k,nks) if n == 1])

    return tidy_d, common_keys

def plot_all(lines, title):

    for fs,dpi,file in [((6,4),150,'foo.png')]: # [((6,4),1200,'foo.pdf') ]: 

        fig, axes = plt.subplots(1,1, squeeze=False
                                 , sharex='all', sharey='row', figsize=fs, dpi=dpi)

        axis = axes[0,0]

        axis.set_title(title)
        axis.set_xlabel('Prob(head) of second coin')
        axis.set_ylabel('Fraction best coin tossed')
        
        for label,line in lines.items():

            xs = np.array([ x for x,y in line ])
            ys = np.array([ y for x,y in line ])

            axis.plot(xs, ys, 'o-', linewidth=0.25, alpha=0.8, label=label)

        axis.legend()
        plt.show()

        fig.savefig(file,bbox_inches='tight')

def pp_warmup(n):
    if n == 0:
        return 'No warmup'
    else:
        return '%d warmup steps' % n
    
def main(argv):

    lines = {}
    
    for file in argv[1:]:

        bits = file.split('-')
        [tag, sn_warmup, sp0, sp1, sn_steps, *rest] = bits

        n_warmup = int(sn_warmup)
        
        p0 = float(sp0)
        p1 = float(sp1)

        n_steps = int(sn_steps)

        with open(file, 'r') as fp:
            ds = json.load(fp)

        for algo in sorted(list(ds.keys()), key=tag_order):
            v = ds[algo]

            f_good_arm = v['f_arm0']
            if p0 > p1:
                f_good_arm = 1.0 - f_good_arm

            k = (tag_name(algo)
                 , pp_warmup(n_warmup)
                 ,'%d steps' % n_steps)

            if k not in lines:
                lines[k] = []

            lines[k].append((p1,f_good_arm))

    lines, common_keys = tidy_keys(lines)

    plot_all(lines, common_keys)
            
if __name__ == "__main__":
    main(sys.argv)    
