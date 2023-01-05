from modnet.preprocessing import MODData
datafeaturized='../../../../../matbench_mp_gap_light_featurized.pkl'
datastructure='../../../../../matbench_mp_gap_matminer.pkl'
import pickle
data=MODData.load(datafeaturized)
data.df_structure = pickle.load(open(datastructure,"rb"))
data.save("matbench_mp_gap.pkl.gz")
