import argparse

import pandas as pd

# Use this code to create a .csv file with the necessary format needed for 
# categorical emotion recognition model

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/data/Odyssey_SER_Challenge/')
args = parser.parse_args()

# Load Original label_consensus.csv file provided with dataset
df = pd.read_csv(args.data_dir + 'Labels/labels_consensus.csv')

# Define the emotions
emotions = ["angry", "sad", "happy", "surprise", "fear", "disgust", "contempt", "neutral"]
emotion_codes = ["A", "S", "H", "U", "F", "D", "C", "N"]

# Create a dictionary for one-hot encoding
# one_hot_dict = {e: [1.0 if e == ec else 0.0 for ec in emotion_codes] for e in emotion_codes}

# Filter out rows with undefined EmoClass
df = df[df['EmoClass'].isin(emotion_codes)]

# # Apply one-hot encoding
# for i, e in enumerate(emotion_codes):
#     df[emotions[i]] = df['EmoClass'].apply(lambda x: one_hot_dict[x][i])

# rename EmoClass to emotion, show emotion category instead of emotion code
df = df.rename(columns={'FileName': 'file', 'EmoClass': 'emotion'})
df['emotion'] = df['emotion'].apply(lambda x: emotions[emotion_codes.index(x)])

# Select relevant columns for the new CSV
df_final = df[['file', 'emotion', 'Split_Set']]

df_train = df_final[df_final.Split_Set == 'Train']
df_train.to_csv('odyssey_train.csv', index=False)

df_dev = df_final[df_final.Split_Set == 'Development']
df_dev.to_csv('odyssey_dev.csv', index=False)

# create test partition from partitions.txt
df_test = pd.read_csv(args.data_dir + 'Partitions.txt', sep='; ', header=None)
df_test = df_test[df_test.iloc[:, 0] == 'Test3']
df_test.columns = ['Split_Set', '']
# rename first column to file
df_test['file'] = df_test['']
df_test.to_csv('odyssey_test.csv', columns=['file', 'Split_Set'], index=False)

# Save the processed data to a new CSV file
# df_final.to_csv('processed_labels.csv', index=False)
