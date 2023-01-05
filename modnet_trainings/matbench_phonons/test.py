
import pickle
import re
df=pickle.load(open("phonons_dfpartial.pkl","rb"))

print(df,df.columns)
rdf_to_del=df.filter(regex='^Radial').columns[50:]
df=df.drop(rdf_to_del,axis=1) 

for col in df.columns:
    if col.startswith("Radial"):
        print(col)
def rdf_rename(x):
    if x.startswith("RadialDistributionFunction"):
        print(x)
        x=x.replace('RadialDistributionFunction|rdf ', 'RadialDistributionFunction|radial distribution function|d_')
        value = re.findall(r'\[(.*)\]A', x)[0].split(' - ')[0][:4]
        print(value)
        x = re.sub(r'\[.*\]A', '', x)
        print(x)
        x = x + value
        print('t',x)
        return x
    else:
        return x
df=df.rename(mapper=rdf_rename, axis='columns')
print(df,df.columns)
for col in df.columns:
    if col.startswith("Radial"):
        print(col)
