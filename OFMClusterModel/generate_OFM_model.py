import sys
# setting path
sys.path.append('../')
from MEGNetCustomTraining import loadMEGNetGroups
import pickle
import numpy as np
import pandas as pd
model_ids=['9b1f667e-e53f-4ad2-b74d-b652b41daddc', '6362d5f0-ee72-43b9-ab25-9eddcc7aafab']
OFMpredictor = loadMEGNetGroups(model_ids,mode='final')
# pickle.dump(MEGNetGroups,open("OFMpredictor.pkl","wb"))


#OFMpredictor=pickle.load(open("OFMpredictor.pkl","rb"))
def predict_from_MEGNetGroup(dataframe, OFMpredictor):
    # have to exclude structures that dont form compatible graphs and their corresponding targets.
    y_preds=[]
    structures=dataframe['structure']
    indexes=dataframe.index
    for i in range(len(OFMpredictor)):
        feats=OFMpredictor[i].group_feats
        model=OFMpredictor[i].model
        scaler=OFMpredictor[i].scaler
        ## structures are converted for each group, though its not necessary
        structures_valid = []
        structures_invalid = []
        for s in structures:
           try:
               graph = model.graph_converter.convert(s)
               structures_valid.append(s)
           except:
               structures_invalid.append(s)
        print(f"Following invalid structures: {structures_invalid}.")
        #structures_valid=np.array(structures_valid)
        y_pred = model.predict_structures(structures_valid)
        y_pred = scaler.inverse_transform(y_pred)
        y_pred = pd.DataFrame(y_pred, columns=feats, index=indexes)
        y_preds.append(y_pred)
    y_preds=pd.concat(y_preds,axis=1)
    return y_preds

from matminer.datasets import load_dataset
perovskite_data=load_dataset('matbench_perovskites')
### data must contain structure column and index is used for result dataframe
predicted_data=predict_from_MEGNetGroup(perovskite_data, OFMpredictor)
print(predicted_data)
pickle.dump(predicted_data, open("result_OFMclustering_perovskites.pkl","wb"))

