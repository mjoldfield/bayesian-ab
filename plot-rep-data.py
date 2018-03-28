import random

import numpy as np
import matplotlib.pyplot as plt

def plot(nsteps, ps):

    fstem = "rnd-{}".format(nsteps)
    
    for fs,dpi,file in [  ((6,4),1200,fstem + '.pdf')
                        , ((6,4), 150,fstem + '.png')]:

        fig, axes = plt.subplots(1,1, squeeze=False
                                 , sharex='all', sharey='row', figsize=fs, dpi=dpi)
        axis = axes[0,0]
        
        axis.set_xlabel('Number of tosses')
        axis.set_ylabel('Number of heads')

        # begin by plotting one trace in order of ascending prob
        # so that the legend looks sane
        pi = 0
        for p in ps:
            plot_p(axis, p, pi, nsteps, True)
            pi += 1

        # now go wild and plot lots of random traces
        for i in range(1000):
            pi = random.randint(0, len(ps)-1)
            p  = ps[pi]
            plot_p(axis, p, pi, nsteps)

        leg = axis.legend()
        for l in leg.legendHandles:
            l.set_linewidth(1.0)
            l.set_alpha(1.0)
        
        fig.savefig(file,bbox_inches='tight')

def plot_p(axis, p, pi, nsteps, label=False):

    ts = np.random.binomial(1, p, nsteps)
    cs = np.cumsum(ts)
            
    line, = axis.plot(cs, linewidth=0.25, alpha=0.5, color="C{}".format(pi))

    if label:
        line.set_label('%.2f' % p)

        
    
def main():
    for n in [1000, 10000]:
        plot(n, [0.01, 0.03, 0.09, 0.1, 0.11])

if __name__ == "__main__":
    main()    
