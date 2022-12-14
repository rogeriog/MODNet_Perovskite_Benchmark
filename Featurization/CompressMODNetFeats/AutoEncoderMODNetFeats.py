import numpy as np
import pickle
import os, sys
sys.path.append('../')
from CompressionFunctions import TestEncoding

def main():
    from modnet.preprocessing import MODData
    data=MODData.load('../../DATAFILES/matbench_perovskites_moddata.pkl.gz')
    Xtoencode=data.df_featurized
    TestEncoding( name_encoder = 'PerovskitesMODNet',
                       dataset = Xtoencode,
               compress_ratios = np.arange(0.9,0.3,-0.1),)
if __name__ == '__main__':
    main()
