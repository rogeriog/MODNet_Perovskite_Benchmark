import sys,os
sys.path.append('../')
from CompressionFunctions import encode_dataset
import pickle
from modnet.preprocessing import MODData
data=MODData.load('../../DATAFILES/matbench_perovskites_moddata.pkl.gz')
Xencoded = encode_dataset(dataset=data.df_featurized,
              scaler='Scaler_PerovskitesMODNet.pkl',
              columns_to_read='encoded_columns.txt',
              autoencoder='PerovskitesMODNet_AutoEncoder_compressratio_1.h5',
              save_name='PerovskitesMODNet_encoded_remap1020.pkl',
              )
import pandas as pd
data.df_featurized = pd.DataFrame(Xencoded,columns=["MatMinerFeats_Compressed|item_"+str(idx) for idx in range(Xencoded.shape[1])], index=data.df_featurized.index)
data.save('matbench_perovskites_moddata_remap1020.pkl.gz')
