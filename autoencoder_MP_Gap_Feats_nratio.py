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
    prefix_name = 'MP_GapFeats'
    savedir='./DATAFILES/'
    for architecture in ['custom_n_b']:
      for bottleneck_ratio in [0.2,0.5,0.8]:
        for loss in ['mse']:
          for batch_size in [64]:
            for epoch in [200]:
              for learning_rate in [0.0005]:
                for custom_n_value in list(np.round(np.arange(1.5,2.6,0.1),2)):
                  # Build and compile the autoencoder model with the current set of hyperparameters
                  if savedir[-1] != '/':
                      savedir+='/'
                  TestEncoding( prefix_name = prefix_name,
                                dataset = Xtoencode,
                                compress_ratio = bottleneck_ratio,
                                architecture=architecture,
                                batch_size=batch_size,
                                epochs = epoch,
                                loss = loss,
                                learning_rate = learning_rate,
                                savedir=savedir+f"{prefix_name}_{architecture}",
                                custom_n_value=custom_n_value,
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
