import pickle
# from modnet import MODData
from modnet.preprocessing import MODData
def get_optimal_feats(test):
    for idx, feat in enumerate(test.optimal_features):
        if feat.startswith("OFM"):
            print(test, idx+1, feat)

test=MODData.load("./train_moddata_f1")
get_optimal_feats(test)
test2=MODData.load("./train_moddata_f2")
get_optimal_feats(test2)
test3=MODData.load("./train_moddata_f3")
get_optimal_feats(test3)
test4=MODData.load("./train_moddata_f4")
get_optimal_feats(test4)
