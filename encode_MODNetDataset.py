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
              scaler='DATAFILES/Scaler_PerovskitesMODNet.pkl',
              columns_to_read='DATAFILES/encoded_columns_MODNetMM.txt',
              autoencoder='DATAFILES/PerovskitesMODNet_AutoEncoders/PerovskitesMODNet_AutoEncoder_compressratio_0.088.h5',
              save_name='PerovskitesMODNet_encoded8.8.pkl',
              feat_prefix="EncodedMatminer"
              )
print('xx',Xencoded)
import pandas as pd
data.df_featurized = pd.DataFrame(Xencoded, index=data.df_featurized.index)
print(data.df_featurized)
data.save('matbench_perovskites_moddata_encoded8.8.pkl.gz')
