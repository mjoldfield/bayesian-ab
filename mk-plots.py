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


def plot_all(dss):

    n_tags = len(dss)
    tags = list(dss.keys())
    
    n = len(dss[tags[0]][0])
    xs = np.array(range(0,n))

    hw = int(n/10)
    if hw < 10:
        hw = 10

    hat_width = hw - 1 + (hw % 2)

    tophat = np.ones(hat_width) / hat_width

    w2 = int((hat_width - 1) / 2)
    cxs = np.array(range(w2,n-w2))

    for fs,dpi,file in [((6,4),1200,'foo.pdf'), ((6,4),150,'foo.png')]:

        fig, axes = plt.subplots(3,n_tags, squeeze=False
                                 , sharex='all', sharey='row', figsize=fs, dpi=dpi)
    
        axes[0,0].set_ylabel('Total Score',   fontsize=7)        
        axes[1,0].set_ylabel('pr(H_2|D)',     fontsize=7)
        axes[2,0].set_ylabel('Average Coin',  fontsize=7)

        col = 0
        for tag in sorted(tags, key=tag_order):
            axes[0, col].set_title(tag_name(tag), fontsize=8)
            axes[1, col].set_autoscaley_on(False)
            axes[1, col].set_ylim([-0.1,1.1])
            axes[2, col].set_autoscaley_on(False)
            axes[2, col].set_ylim([-0.1,1.1])

            for i in [0,1,2]:
                axes[i,col].tick_params(axis='both', which='both', labelsize=6)

            for ds in dss[tag]:
                plot_thing(axes[0,col],  xs, ds, 'score',  None,   None)
                plot_thing(axes[1,col], cxs, ds, 'pr(h2)', tophat, 0.01)
                plot_thing(axes[2,col], cxs, ds, 'arm',    tophat, 0.01)
            
            col += 1

        #plt.show()

        fig.savefig(file,bbox_inches='tight')

def plot_thing(axis, xs, ds, k, smooth, jitter):
    if k not in ds[0]:
        return
    
    ys = np.array([ d[k] for d in ds ])
    
    if smooth is not None:
        ys = np.convolve(ys, smooth, mode='valid')

    if jitter is not None:
        jits = np.random.normal(0, jitter, ys.shape)
        ys += jits
        
    axis.plot(xs, ys, linewidth=0.25, alpha=0.3)
    
def main(argv):
    dss = {}
    
    for file in argv[1:]:
        bits = file.split('-')
        tag = bits[-3]

        if tag not in dss:
            dss[tag] = []

        with open(file, 'r') as fp:
            dss[tag].append(json.load(fp))

    plot_all(dss)

if __name__ == "__main__":
    main(sys.argv)    
