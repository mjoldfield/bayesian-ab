import sys
import json
import re

import numpy as np
import matplotlib.pyplot as plt

def plot_all(dss):

    n_tags = len(dss)
    tags = list(dss.keys())
    
    n = len(dss[tags[0]][0])
    xs = np.array(range(0,n))

    hw = int(n/100)
    if hw < 10:
        hw = 10

    hat_width = hw - 1 + (hw % 2)

    print("%d => %d => %d" % (n, hw, hat_width))
    
    tophat = np.ones(hat_width) / hat_width

    w2 = int((hat_width - 1) / 2)
    cxs = np.array(range(w2,n-w2))
       
    fig, axes = plt.subplots(3,n_tags, squeeze=False
                             , sharex='all', sharey='row', figsize=(6,4), dpi=300)
    

    axes[0,0].set_ylabel('Total',         fontsize=10)
    axes[1,0].set_ylabel('Average Score', fontsize=8)
    axes[2,0].set_ylabel('Average Coin',  fontsize=12)
    
    col = 0
    for tag in tags:
        axes[0, col].set_title(tag)
        
        for ds in dss[tag]:
            ss = np.array([ d['score'] for d in ds ])
            axes[0,col].plot(xs, ss, linewidth=0.5)

            ts = np.array([ d['draw'] for d in ds ])
            axes[1,col].plot(cxs, np.convolve(ts, tophat, mode='valid'), linewidth=0.5)

            us = np.array([ d['arm'] for d in ds ])
            axes[2,col].plot(cxs, np.convolve(us, tophat, mode='valid'), linewidth=0.5)
        col += 1

    plt.show()

    fig.savefig('foo.pdf',bbox_inches='tight')
    #fig.savefig('foo.png',bbox_inches='tight')
    
def main(argv):
    dss = {}
    
    for file in argv[1:]:
        bits = file.split('-')
        tag = bits[-2]

        if tag not in dss:
            dss[tag] = []

        with open(file, 'r') as fp:
            dss[tag].append(json.load(fp))

    plot_all(dss)

if __name__ == "__main__":
    main(sys.argv)    
