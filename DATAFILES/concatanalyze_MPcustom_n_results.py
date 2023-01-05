import pandas as pd

file_list=[f'MP_GapFeats{i}_custom_n_b/MP_GapFeats{i}_EncoderResults.txt' for i in ['','2','3']]
# Create an empty list to store the dataframes
df_list = []
# Loop through a list of file names
names='architecture|           loss_fn|        batch_size|            epochs|     learning_rate|n_bottleneck_ratio|      n_bottleneck|        train_loss|          val_loss|       correlation|       cosine dist|               MAE|              RMSE|                R2|  RMSE zero-vector'.split('|')
names=[x.strip() for x in names]
print(names)

for file in file_list:
    # Read the file and skip the first two rows
    df = pd.read_csv(file, sep='\s+',names=names, skiprows=2, comment='#')
    # Append the dataframe to the list
    df_list.append(df)
# Concatenate all the dataframes in the list
df = pd.concat(df_list, ignore_index=True)
# Write the concatenated tables to a new .txt file
df['neurons'] = pd.to_numeric(df['architecture'].str.slice(0, 3))
neurons_list=list(set(df['neurons'].values))
neurons_list=sorted([round(float(x),2) for x in neurons_list])
for neurons_i in neurons_list:
    filtered_df = df.loc[df['neurons'] == neurons_i]
    print(neurons_i)
    print(filtered_df['MAE'].mean())

"""
df.to_csv('MP_GapFeats_CustomN_b.txt', index=False, sep=' ')
print(df)
print(df['n_bottleneck_ratio'])
"""
