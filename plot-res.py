import sys
import json
import re

import argparse

import numpy as np
import matplotlib.pyplot as plt

def tag_info(tag):
    i = { 'aeg':   (50, "AEG")
        , 'ucb1':  (60, "UCB 1")
        , 'bayes': (10,  "Bayes")
        , 'rnd':   (20,  "Bayes Rnd")
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

def plot_all(lines, title, outstem):

    for fs,dpi,suffix in [((6,4),150,'png'), ((6,4),1200,'pdf') ]: 

        fig, axes = plt.subplots(1,1, squeeze=False
                                 , sharex='all', sharey='row', figsize=fs, dpi=dpi)

        axis = axes[0,0]

        axis.set_title(title)
        axis.set_xlabel('Prob(head) of second coin')
        axis.set_ylabel('Fraction best coin tossed')
        
        for label,line in lines.items():

            xs = np.array([ x for x,y in line ])
            ys = np.array([ y for x,y in line ])

            axis.plot(xs, ys, 'o-', linewidth=1.0, alpha=0.5, label=label)

        axis.legend()
#        plt.show()

        file = outstem + '.' + suffix
        fig.savefig(file,bbox_inches='tight')

def dump_table(lines, title, outstem):

    n_algos = len(lines)
    
    table = "table(spaced).\n"    

    table += "_|/2. Algorithm|\%d. Probability| \\\n" % n_algos

    line0 = list(lines.values())[0]
    table += '|'.join([ '_' ] + [ '%0.2f' % pr for pr,y in line0 ] + [ " \\\n" ])
    
    for label,line in lines.items():
        table += '|'.join([ '', '_. ' + label ] + [ '%0.3f' % y for pr,y in line ] + [ " \\\n" ])

    print(table)

def pp_warmup(n):
    if n == 0:
        return 'No warmup'
    else:
        return '%d warmup steps' % n
    
def main(files, algos, outstem, title):

    lines = {}
    
    for file in files:

        bits = file.split('-')
        [tag, sn_warmup, sp0, sp1, sn_steps, *rest] = bits

        n_warmup = int(sn_warmup)
        
        p0 = float(sp0)
        p1 = float(sp1)

        n_steps = int(sn_steps)

        with open(file, 'r') as fp:
            ds = json.load(fp)

        for algo in sorted(list(ds.keys()), key=tag_order):

            if algos is not None and algo not in algos:
                continue

            v = ds[algo]

            f_good_arm = v['f_good_arm']

            k = (tag_name(algo)
                 , pp_warmup(n_warmup)
                 ,'%d steps' % n_steps)

            if k not in lines:
                lines[k] = []

            lines[k].append((p1,f_good_arm))

    lines, common_keys = tidy_keys(lines)

    if title is None:
        title = common_keys

    plot_all(lines, title, outstem)

    dump_table(lines, title, outstem)
            
if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument(dest='filenames', metavar='filename', nargs='*')

    p.add_argument('--algo', metavar='algorithm', required=False,
                   dest='algos', action='append')

    p.add_argument('-o', '--outstem', required=True
                   , action='store', dest='outstem')

    p.add_argument('-t', '--title',  required=False
                   , action='store', dest='title')

    args = p.parse_args()

    main(args.filenames, args.algos, args.outstem, args.title)    
