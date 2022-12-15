from modnet.preprocessing import MODData 
data=MODData.load('matbench_perovskites_moddata.pkl.gz')
#print(data)
datatmp=data.df_featurized
datatmp['e_form']=data.df_targets
datatmp['structure']=data.df_structure
datatmp=datatmp.sample(1000,random_state=1)
# print(datatmp)
data.df_featurized=datatmp.drop(['e_form','structure'],axis=1)
data.df_targets=datatmp[['e_form']]
data.df_structure=datatmp[['structure']]
data.save('matbench_perovskites_moddata1000.pkl.gz')
