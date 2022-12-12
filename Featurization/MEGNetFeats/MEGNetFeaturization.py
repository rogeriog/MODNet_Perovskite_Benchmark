from megnet.utils.models import load_model, AVAILABLE_MODELS
import numpy as np
from keras.models import Model
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
# print(AVAILABLE_MODELS)
def get_MEGNetFeaturesDF(structures):
    MEGNetFeats_structs=[]
    for model_name in ['Eform_MP_2019','Efermi_MP_2019','Bandgap_MP_2018','logK_MP_2019','logG_MP_2019']:
        model=load_model(model_name) 
        intermediate_layer_model = Model(inputs=model.input,
                             outputs=model.layers[-3].output)   
        MEGNetModel_structs=[]
        for s in structures:
            try:
                graph = model.graph_converter.convert(s)
                inp = model.graph_converter.graph_to_input(graph)
                pred = intermediate_layer_model.predict(inp, verbose=False)
                model_struct=pd.DataFrame([pred[0][0]], 
                                          columns=[f"MEGNet_{model_name}_{idx+1}" for idx in 
                                                   range(len(pred[0][0]))])
                MEGNetModel_structs.append(model_struct)
            except Exception as e:
                print(e)
                print("Probably an invalid structure was passed to the model, continuing..")
                model_struct=pd.DataFrame([np.nan]*32, 
                                          columns=[f"MEGNet_{model_name}_{idx+1}" for idx in 
                                                   range(len(pred[0][0]))])
                continue
        ## now append the columns with the layer of each model
        MEGNetModel_structs=pd.concat(MEGNetModel_structs,axis=0)
        MEGNetFeats_structs.append(MEGNetModel_structs)
        print(f"Features calculated for model {model_name}.")
    ## now every structure calculated with each model is combined in a final dataframe
    MEGNetFeats_structs=pd.concat(MEGNetFeats_structs,axis=1)
    return MEGNetFeats_structs

from modnet.preprocessing import MODData
data=MODData.load('../../DATAFILES/matbench_perovskites_moddata.pkl.gz')
import pickle
structures=data.df_structure['structure']
slices=list(range(0,len(structures),1000))+[None]
for idx in range(len(slices)-1):
    if idx < 4 : 
        continue
    print(f"Processing slice {idx+1} out of {len(slices)}")
    MEGNetFeats_struct=get_MEGNetFeaturesDF(structures[slices[idx]:slices[idx+1]])
    pickle.dump(MEGNetFeats_struct,open(f"MEGNetFeats_struct_slice{idx}.pkl", "wb"))
    del MEGNetFeats_struct ## free memory
