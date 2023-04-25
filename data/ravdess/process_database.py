# adapted from https://www.kaggle.com/code/shivamburnwal/speech-emotion-recognition
 
import os
import pandas as pd

# ravdess source directory
source_dir = './'
database_name = 'ravdess'

ravdess_directory_list = os.listdir(Ravdess)

file_emotion = []
file_speaker = []
file_gender = []
file_path = []
for dir in ravdess_directory_list:
    # as their are 20 different actors in our previous directory we need to extract files for each actor.
    if os.path.isdir(source_dir + dir):
        actor = os.listdir(source_dir + dir)
        for file in actor:
            part = file.split('.')[0]
            part = part.split('-')
            # third part in each file represents the emotion associated to that file.
            file_emotion.append(int(part[2]))
            file_gender.append(int(part[6]))
            file_speaker.append( f'{database_name}_{str(part[6])}')
            file_path.append(source_dir + dir + '/' + file)
        
# dataframe for emotion of files
#emotion_df = pd.DataFrame(file_emotion, columns=['emotion'])
data_df = pd.DataFrame(list(zip(file_emotion,file_speaker)), columns=['emotion', 'speaker'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['file'])
Ravdess_df = pd.concat([data_df, path_df], axis=1)

# changing integers to actual emotions.
Ravdess_df.emotion.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)
Ravdess_df = Ravdess_df.set_index('file')
print(Ravdess_df.head())
Ravdess_df.to_csv(database_name+'.csv')