import numpy as np
import pickle
import os, sys
sys.path.append('../')
from CompressionFunctions import TestEncoding

def main():
    OFMfeaturized=pickle.load(open('../OFM_featurization/OFM_featurizedDF.pkl','rb'))
    TestEncoding( name_encoder = 'PerovskitesOFM',
                       dataset = OFMfeaturized,
               compress_ratios = np.arange(0.9,0.3,-0.1),)
if __name__ == '__main__':
    main()
