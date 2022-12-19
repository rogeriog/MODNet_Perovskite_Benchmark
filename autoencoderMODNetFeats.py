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
    TestEncoding( name_encoder = 'PerovskitesMODNet_2n_n_b',
                       dataset = Xtoencode,
               compress_ratios = np.arange(0.9,0,-0.05),
               savedir='./DATAFILES/CompressMODNet_2n_n_b',               
               mode='2n_n_b',
               )
    TestEncoding( name_encoder = 'PerovskitesMODNet_n_2n_b',
                       dataset = Xtoencode,
               compress_ratios = np.arange(0.9,0,-0.05),
               savedir='./DATAFILES/CompressMODNet_n_2n_b',
               # epochs=20, 
               mode='n_2n_b',
               )
"""
        elif mode == '3n_b':
            n_bottleneck=int(n_inputs*n_bottleneck_ratio)
            layers_structure= [3*n_inputs, int(n_bottleneck) ]
            model = create_autoencoder(input_shape=n_inputs, layers_structure=layers_structure)
        elif mode == 'n_b':
            n_bottleneck=int(n_inputs*n_bottleneck_ratio)
            layers_structure= [n_inputs, int(n_bottleneck) ]
            model = create_autoencoder(input_shape=n_input
"""
if __name__ == '__main__':
    main()
