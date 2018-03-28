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
        
        xs = np.linspace(1, nsteps, num=nsteps)

        for i in range(1000):

            pi = random.randint(0, len(ps)-1)
            p  = ps[pi]

            ts = np.random.binomial(1, p, nsteps)
            cs = np.cumsum(ts)
            
            axis.plot(cs, linewidth=0.25, alpha=0.5, color="C{}".format(pi))

        fig.savefig(file,bbox_inches='tight')

    
def main():
    for n in [1000, 10000]:
        plot(n, [0.01, 0.03, 0.09, 0.1, 0.11, 0.3])

if __name__ == "__main__":
    main()    
