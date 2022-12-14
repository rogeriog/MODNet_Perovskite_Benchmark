import sys,os
sys.path.append('../')
from CompressionFunctions import encode_dataset
import pickle
OFMfeaturized=pickle.load(open('../OFM_featurization/OFM_featurizedDF.pkl','rb'))
encode_dataset(dataset=OFMfeaturized,
              scaler='ScalerOFMPerovskites_MinMax_testsize0.1_random1.pkl',
              columns_to_read='columns_encoded.txt',
              autoencoder='PerovskitesOFM_AutoEncoder_compressratio_0.12.h5',
              save_name='PerovskitesOFM_encoded43.pkl',
              )
