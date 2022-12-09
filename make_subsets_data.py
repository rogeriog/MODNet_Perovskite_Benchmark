import os, shutil 
from modnet.preprocessing import MODData 

def initialize_data():
    ### the full set calculations files
    shutil.copyfile('./DATAFILES/matbench_perovskites_moddata.pkl.gz', './noOFM/fullset/matbench_perovskites/precomputed/matbench_perovskites_moddata.pkl.gz')
    shutil.copyfile('./DATAFILES/matbench_perovskites_moddataOFM.pkl.gz', 'withOFM/fullset/matbench_perovskites/precomputed/matbench_perovskites_moddataOFM.pkl.gz')
    ### calculation on the subsets
    for n_samples in [1000, 5000]:
        data=MODData.load('./DATAFILES/matbench_perovskites_moddata.pkl.gz')
        data_OFM=MODData.load('./DATAFILES/matbench_perovskites_moddataOFM.pkl.gz')
        print(data.df_featurized)
        print(data_OFM.df_featurized)
        datatmp=data.df_featurized
        datatmp['e_form']=data.df_targets
        datatmp['structure']=data.df_structure
        datatmp=datatmp.sample(n_samples,random_state=1)
        print(datatmp)
        #datatmp=datatmp.sample(5000,random_state=1)
        datatmp_OFM=data_OFM.df_featurized
        datatmp_OFM['e_form']=data_OFM.df_targets
        datatmp_OFM['structure']=data_OFM.df_structure
        datatmp_OFM=datatmp_OFM.sample(n_samples,random_state=1)
        print(datatmp_OFM)
        ## substitute in MODData and save
        data.df_featurized=datatmp.drop(['e_form','structure'],axis=1)
        data.df_targets=datatmp[['e_form']]
        data.df_structure=datatmp[['structure']]
        data_OFM.df_featurized=datatmp_OFM.drop(['e_form','structure'],axis=1)
        data_OFM.df_targets=datatmp_OFM[['e_form']]
        data_OFM.df_structure=datatmp_OFM[['structure']]
        if n_samples == 1000:
            data.save('noOFM/subset1k/matbench_perovskites/precomputed/matbench_perovskites_moddata1000.pkl.gz')
            data_OFM.save('withOFM/subset1k/matbench_perovskites/precomputed/matbench_perovskites_moddataOFM1000.pkl.gz')
        elif n_samples == 5000: 
            data.save('noOFM/subset5k/matbench_perovskites/precomputed/matbench_perovskites_moddata5000.pkl.gz')
            data_OFM.save('withOFM/subset5k/matbench_perovskites/precomputed/matbench_perovskites_moddataOFM5000.pkl.gz')

def initialize_dirs():
    types=["withOFM","noOFM"]
    subfolders=["subset1k","subset5k","fullset"]
    matbench_folders=["final_model","folds","plots","precomputed","results"]
    for folder in types:
        try:
            os.mkdir(folder)
        except OSError:
            print("Folder already created.")
            continue
        for subfolder in subfolders:
            try:
                os.mkdir(folder+'/'+subfolder)
            except OSError:
                print("Folder already created.")
                continue
            shutil.copyfile("run_benchmark.py", folder+'/'+subfolder+"/run_benchmark.py")
            for matbench_folder in matbench_folders:
                try:
                    os.mkdir(folder+'/'+subfolder+'/'+'matbench_perovskites/')
                except OSError:
                    print("Folder already created.")
                try:
                    os.mkdir(folder+'/'+subfolder+'/'+'matbench_perovskites/'+matbench_folder) 
                except OSError:
                    print("Folder already created.")
                    continue
if __name__ == "__main__":
    initialize_dirs()
    initialize_data()
