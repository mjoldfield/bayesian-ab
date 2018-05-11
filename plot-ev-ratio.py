import numpy as np
import scipy.special     as sc
import matplotlib.pyplot as plt


def plot(n):

    (xs, ys, zs) = do_sum(n,n)
    
    fstem = "evratio-{}".format(n)

    levels = [0.0, 0.1, 0.5, 0.9, 1.0]
    
    for fs,dpi,file in [  ((6,4),1200,fstem + '.pdf')
                        , ((6,4), 150,fstem + '.png')]:

        fig, axes = plt.subplots(1,1, squeeze=False
                                 , sharex='all', sharey='row', figsize=fs, dpi=dpi)
        axis = axes[0,0]

        axis.set_title(r'$\mathrm{pr}(\mathscr{H}_2|D)$');
        axis.set_xlabel('Coin A: number of heads')
        axis.set_ylabel('Coin B: number of heads')

        cs = axis.imshow(zs, extent=[0,n,0,n], origin='lower', cmap=plt.get_cmap('PRGn'))
        plt.colorbar(cs)
                
        fig.savefig(file,bbox_inches='tight')

def do_sum(na,nb):
    x = np.linspace(0, na)
    y = np.linspace(0, nb)

    xg, yg = np.meshgrid(x, y)

    zg = evRatio(na, nb, xg, yg)

    return (xg, yg, zg)

def log_binomial(h, k):
    return sc.gammaln(h + 1) - sc.gammaln(k + 1) - sc.gammaln(h - k + 1)

def evRatio(na, nb, ka, kb):

    log_ev1 = (log_binomial(na, ka) 
               + log_binomial(nb,kb) 
               - log_binomial(na + nb, ka + kb) 
	             - np.log(na + nb + 1) 
              ) 

    log_ev2 = -(np.log(1 + na) + np.log(1 + nb))

    log_ratio = log_ev2 - log_ev1
    pr_h2 = 1.0 / (1.0 + np.exp(-log_ratio))
    
    return pr_h2

        
    
def main():
    for n in [100]:
        plot(n)

if __name__ == "__main__":
    main()    
