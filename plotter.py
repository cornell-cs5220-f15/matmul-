#!/share/apps/python/anaconda/bin/python


import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def make_plot(dir):
    # Plot results of timing trials
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir) 
    for file in os.listdir(path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(path,file))
            plt.plot(df['size'], df['mflop']/1e3, label=file[:-3])
    plt.xlabel('Dimension')
    plt.ylabel('Gflop/s')

def show(runs):
    "Show plot of timing runs (for interactive use)"
    make_plot(runs)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def main(dir):
    "Show plot of timing runs (non-interactive)"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir) 
    make_plot(dir)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(os.path.join(path,'timing.pdf'), bbox_extra_artists=(lgd,), bbox_inches='tight')

if __name__ == "__main__":
    main(sys.argv[1])
