import sys
import json

import numpy as np
import matplotlib.pyplot as plt

def plot_all(dss):

    xs = np.array(range(0,len(dss[0])))

    hat_width = 9
    tophat = np.ones(hat_width) / hat_width

    w2 = int((hat_width - 1) / 2)
    cxs = np.array(range(w2,len(dss[0])-w2))
    
    fig, (score, delta, arm) = plt.subplots(3,1)

    for ds in dss:
        ss = np.array([ d['score'] for d in ds ])
        score.plot(xs, ss)

        ts = np.array([ d['draw'] for d in ds ])
        delta.plot(cxs, np.convolve(ts, tophat, mode='valid'))

        us = np.array([ d['arm'] for d in ds ])
        arm.plot(cxs, np.convolve(us, tophat, mode='valid'))


        
    plt.show()
    
def main(argv):
    dss = []
    
    for file in argv[1:]:
        with open(file, 'r') as fp:
            dss.append(json.load(fp))

    plot_all(dss)

if __name__ == "__main__":
    main(sys.argv)    
