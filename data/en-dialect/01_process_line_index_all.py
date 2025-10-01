import numpy as np
import pandas as pd
import io

#add headers directly
df = pd.read_csv('./line_index_all.csv', 
                 header=None,  
                 names=['line_ID', 'audio_with_out_wav', 'transcription']) 

print(df.columns)

# extract speaker, label(dialect type) and gender information from csv file 
df['speaker'] = df['audio_with_out_wav'].str.split('_').str[:2].str.join('_')

df['dialect'] = df['audio_with_out_wav'].str.split('_').str[0].str[:-1]

df['gender'] = df['audio_with_out_wav'].str.split('_').str[0].str[-1]

# 顯示處理後的 DataFrame
print(df)

#save new csv
df.to_csv('./line_index_all_process_1.csv', index=False)