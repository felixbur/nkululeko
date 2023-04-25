# adapted from https://www.kaggle.com/code/shivamburnwal/speech-emotion-recognition
 
import os
import pandas as pd

# ravdess source directory
Ravdess = './'

ravdess_directory_list = os.listdir(Ravdess)

file_emotion = []
file_path = []
for dir in ravdess_directory_list:
    # as their are 20 different actors in our previous directory we need to extract files for each actor.
    if os.path.isdir(Ravdess + dir):
        actor = os.listdir(Ravdess + dir)
        for file in actor:
            part = file.split('.')[0]
            part = part.split('-')
            # third part in each file represents the emotion associated to that file.
            file_emotion.append(int(part[2]))
            file_path.append(Ravdess + dir + '/' + file)
        
# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['emotion'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['file'])
Ravdess_df = pd.concat([emotion_df, path_df], axis=1)

# changing integers to actual emotions.
Ravdess_df.emotion.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)
Ravdess_df = Ravdess_df.set_index('file')
print(Ravdess_df.head())
Ravdess_df.to_csv('ravdess.csv')