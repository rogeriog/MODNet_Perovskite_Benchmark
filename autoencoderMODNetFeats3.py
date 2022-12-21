import numpy as np
import pickle
import os, sys
#sys.path.append('../')
#from CompressionFunctions import TestEncoding
from Featurization.CompressionFunctions import TestEncoding
def main():
    from modnet.preprocessing import MODData
    data=MODData.load('../matbench_mp_gap_light_featurized.pkl')
    Xtoencode=data.df_featurized.fillna(-1)
    print(Xtoencode)
    TestEncoding( name_encoder = 'MPGap_MODNet_3n_2n_b',
                  dataset = Xtoencode,
                  compress_ratios = np.arange(1,0,-0.05),
                  savedir='./DATAFILES/CompressMODNet_3n_2n_b_MPGap',
                  # epochs=20, 
                  mode='3n_2n_b',
               )
    TestEncoding( name_encoder = 'MPGap_MODNet_2n_3n_b',
                       dataset = Xtoencode,
               compress_ratios = np.arange(1,0,-0.05),
               savedir='./DATAFILES/CompressMODNet_2n_3n_b_MPGap',               
               mode='2n_3n_b',
               )
def main_perovskites():
    from modnet.preprocessing import MODData
    data=MODData.load('./DATAFILES/matbench_perovskites_moddata.pkl.gz')
    Xtoencode=data.df_featurized
    print(Xtoencode)
    TestEncoding( name_encoder = 'PerovskitesMODNet_3n_2n_b',
                       dataset = Xtoencode,
               compress_ratios = np.arange(0.9,0,-0.05),
               savedir='./DATAFILES/CompressMODNet_3n_2n_b',               
               mode='3n_2n_b',
               )
    TestEncoding( name_encoder = 'PerovskitesMODNet_2n_3n_b',
                       dataset = Xtoencode,
               compress_ratios = np.arange(0.9,0,-0.05),
               savedir='./DATAFILES/CompressMODNet_2n_3n_b',
               # epochs=20, 
               mode='2n_3n_b',
               )
if __name__ == '__main__':
    main()
