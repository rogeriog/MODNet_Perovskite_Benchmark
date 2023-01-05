import numpy as np
import pickle
import os, sys
#sys.path.append('../')
#from CompressionFunctions import TestEncoding
from Featurization.CompressionFunctions import HyperParameterTestEncoding, TestEncoding


def main():
    import os
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    from modnet.preprocessing import MODData
    data=MODData.load('../matbench_mp_gap_light_featurized.pkl')
    Xtoencode=data.df_featurized.fillna(-1)
    print(Xtoencode)
    HyperParameterTestEncoding( dataset = Xtoencode,
        prefix_name = 'MP_GapFeats',
        bottleneck_ratios = [0.5], 
        batch_sizes = [16,32,64],
        epochs_list = [100,200,300],
        loss_functions = ['mse'],
        learning_rates = [0.0005, 0.001, 0.002],
        architectures = ['2n_conv2_b'], 
        savedir = './DATAFILES/',
        )
def main_perovskites():
    from modnet.preprocessing import MODData
    data=MODData.load('./DATAFILES/matbench_perovskites_moddata.pkl.gz')
    Xtoencode=data.df_featurized
    print(Xtoencode)
    TestEncoding( name_encoder = 'PerovskitesMODNet_2n_b',
                       dataset = Xtoencode,
               compress_ratios = np.arange(0.9,0,-0.05),
               savedir='./DATAFILES/CompressMODNet_2n_b',
               # epochs=20, 
               mode='default',
               )
    TestEncoding( name_encoder = 'PerovskitesMODNet_2n_m2nb_b',
                       dataset = Xtoencode,
               compress_ratios = np.arange(0.9,0,-0.05),
               savedir='./DATAFILES/CompressMODNet_2n_m2nb_b',               
               mode='2n_m2nb_b',
               )
if __name__ == '__main__':
    main()
