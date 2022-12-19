from CompressionFunctions import encode_dataset
import pickle
X=pickle.load(open("../DATAFILES/OFM_featurization/OFM_featurizedDF.pkl","rb"))
#encode_dataset(X,
#        scaler = './CompressOFM/ScalerOFMPerovskites_MinMax_testsize0.1_random1.pkl',
#        columns_to_read = './CompressOFM/encoded_columns.txt',
#        autoencoder = './CompressOFM/PerovskitesOFM_AutoEncoder_compressratio_0.5.h5',
#        save_name = '../DATAFILES/PerovskitesOFM_0.5compress.pkl',
#        feat_prefix = 'OFMencoded50'
#        )
encode_dataset(X,
        scaler = './CompressOFM/ScalerOFMPerovskites_MinMax_testsize0.1_random1.pkl',
        columns_to_read = './CompressOFM/encoded_columns.txt',
        autoencoder = './CompressOFM/PerovskitesOFM_AutoEncoder_compressratio_0.16.h5',
        save_name = '../DATAFILES/PerovskitesOFM_0.16compress.pkl',
        feat_prefix = 'OFMencoded16'
        )
