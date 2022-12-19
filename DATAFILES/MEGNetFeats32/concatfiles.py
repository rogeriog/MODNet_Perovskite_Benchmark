import glob, pickle
import pandas as pd
files=glob.glob("MEGNetFeats_struct_slice*")
tmpDFs=[0 for idx in files]
for idx in range(len(files)):
    tmpDFs[idx]=pickle.load(open(f"MEGNetFeats_struct_slice{idx}.pkl","rb"))
FinalDF=pd.concat(tmpDFs,axis=0)
pickle.dump(FinalDF,open("MEGNetFeats32_Perovsk.pkl","wb"))
