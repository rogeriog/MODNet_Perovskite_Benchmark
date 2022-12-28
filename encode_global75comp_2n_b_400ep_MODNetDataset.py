import sys,os
sys.path.append('./Featurization/')
from CompressionFunctions import encode_dataset
import pickle
from modnet.preprocessing import MODData
data=MODData.load('./DATAFILES/matbench_perovskites_moddata.pkl.gz')
#Xencoded = encode_dataset(dataset=data.df_featurized,
#              scaler='Scaler_PerovskitesMODNet.pkl',
#              columns_to_read='encoded_columns.txt',
#              autoencoder='PerovskitesMODNet_AutoEncoder_compressratio_1.h5',
#              save_name='PerovskitesMODNet_encoded_remap1020.pkl',
#              )
Xencoded = encode_dataset(dataset=data.df_featurized,
              scaler='DATAFILES/dump/CompressMODNet_2n_b_MPGap/Scaler_MPGap_MODNet_2n_b.pkl',
              columns_to_read='DATAFILES/dump/CompressMODNet_2n_b_MPGap/encoded_columns.txt',
              autoencoder='DATAFILES/dump/CompressMODNet_2n_b_MPGap/MPGap_MODNet_2n_b_AutoEncoder_compressratio_0.75.h5',
              save_name='PerovskitesMODNet_GlobalEncoded75.pkl',
              feat_prefix="GlobalEncodedMatminer"
              )
#print('xx',Xencoded)
#import pandas as pd
#data.df_featurized = pd.DataFrame(Xencoded, index=data.df_featurized.index)
#print(data.df_featurized)
#data.save('matbench_perovskites_moddata_GlobalEncoded75.pkl.gz')
