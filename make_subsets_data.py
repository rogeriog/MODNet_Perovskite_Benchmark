import os, shutil 
from modnet.preprocessing import MODData 
def replace_line(file_name, line_num, text):
    lines = open(file_name, 'r').readlines()
    lines[line_num] = text
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()

def initialize_data(path_full_data,dirnames):
    for dirname in dirnames:
    ### the full set calculations files
        shutil.copyfile(path_full_data, './'+dirname+'/fullset/matbench_perovskites/precomputed/'+path_full_data.split('/')[-1])
        ### calculation on the subsets
        for n_samples in [1000, 5000]:
            data=MODData.load(path_full_data)
            print(data.df_featurized)
            datatmp=data.df_featurized
            datatmp['e_form']=data.df_targets
            datatmp['structure']=data.df_structure
            datatmp=datatmp.sample(n_samples,random_state=1)
            print(datatmp)
            ## substitute in MODData and save
            data.df_featurized=datatmp.drop(['e_form','structure'],axis=1)
            data.df_targets=datatmp[['e_form']]
            data.df_structure=datatmp[['structure']]
            if n_samples == 1000:
                data.save(dirname+'/subset1k/matbench_perovskites/precomputed/'+path_full_data.split('/')[-1].split('.')[0]+'1000.pkl.gz')
            elif n_samples == 5000: 
                data.save(dirname+'/subset5k/matbench_perovskites/precomputed/'+path_full_data.split('/')[-1].split('.')[0]+'5000.pkl.gz')

def initialize_dirs(dirnames):
    # types=["MODNetCompressed"]
    subfolders=["subset1k","subset5k","fullset"]
    matbench_folders=["final_model","folds","plots","precomputed","results"]
    for folder in dirnames:
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
            shutil.copyfile("submit.sh", folder+'/'+subfolder+"/submit.sh")
            replace_line(folder+'/'+subfolder+"/submit.sh",1,f'#SBATCH --job-name={folder}{subfolder}\n')
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
    initialize_dirs(["MODNetRemaped"])
    #initialize_data('./DATAFILES/matbench_perovskites_moddata.pkl.gz',["MODNetCompressed"])
    #initialize_data('./Featurization/CompressMODNetFeats/matbench_perovskites_moddata_compressed244.pkl.gz',["MODNetCompressed"])
    initialize_data('./Featurization/CompressMODNetFeats/matbench_perovskites_moddata_remap1020.pkl.gz',["MODNetRemaped"])
