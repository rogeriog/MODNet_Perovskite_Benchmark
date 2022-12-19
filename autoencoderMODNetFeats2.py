import numpy as np
import pickle
import os, sys
#sys.path.append('../')
#from CompressionFunctions import TestEncoding
from Featurization.CompressionFunctions import TestEncoding
def main():
    from modnet.preprocessing import MODData
    data=MODData.load('./DATAFILES/matbench_perovskites_moddata.pkl.gz')
    Xtoencode=data.df_featurized
    print(Xtoencode)
    TestEncoding( name_encoder = 'PerovskitesMODNet_3n_b',
                       dataset = Xtoencode,
               compress_ratios = np.arange(0.9,0,-0.05),
               savedir='./DATAFILES/CompressMODNet_3n_b',               
               mode='3n_b',
               )
    TestEncoding( name_encoder = 'PerovskitesMODNet_n_b',
                       dataset = Xtoencode,
               compress_ratios = np.arange(0.9,0,-0.05),
               savedir='./DATAFILES/CompressMODNet_n_b',
               # epochs=20, 
               mode='n_b',
               )
if __name__ == '__main__':
    main()
