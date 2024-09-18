"""
code by Anna Derington

"""
import json
import os
import shutil

import audeer
import audformat
import numpy as np
import pandas as pd


def main():
    description = (
        'Emozionalmente is an extensive, crowdsourced Italian emotional speech corpus. '
        'The dataset consists of 6902 labeled samples acted out by 431 amateur actors '
        'while verbalizing 18 different sentences expressing the Big Six emotions '
        '(anger, disgust, fear, joy, sadness, surprise) plus neutrality. '
        'Labels represent the emotional communicative intention of the actors '
        '(i.e., the seven emotional states). '
        'Recordings were generally obtained with non-professional equipment. '
        'They are .wav files, mono-channel, and have a sample size of 16 bits and a sample rate of 16000 Hz. '
        'Each audio recording lasts 3.81 seconds (SD = 0.99 seconds). '
        'The emotional content of the clips were validated by asking 829 humans (5 evaluations per audio) '
        'to guess the emotion contained in each recording. Humans obtained a general accuracy of 66%. '
    )
    languages = ['it']
    db = audformat.Database(
        name='Emozionalmente',
        source='https://zenodo.org/record/6569824#.ZGsf1hlByV4',
        usage=audformat.define.Usage.COMMERCIAL,
        description=description,
        license=audformat.define.License.CC_BY_4_0,
        languages=languages,
        author='Fabio Catania',
    )
    current_dir = os.path.dirname(__file__)
    build_dir = audeer.mkdir(os.path.join(current_dir, './build'))
    source_dir = 'emozionalmente_dataset'
    audio_dir_name = 'audio'
    audio_path = os.path.join(current_dir, source_dir, audio_dir_name)

    user_path = os.path.join(current_dir, source_dir, 'metadata', 'users.csv')
    samples_path = os.path.join(current_dir, source_dir, 'metadata', 'samples.csv')
    evaluations_path = os.path.join(current_dir, source_dir, 'metadata', 'evaluations.csv')

    emotion_labels = {
        'anger': {'original': 'anger'},
        'disgust': {'original': 'disgust'},
        'fear': {'original': 'fear'},
        'happiness': {'original':'joy'},
        'neutral': {'original': 'neutrality'},
        'no_agreement': {'original': 'no_agreement'},
        'sadness': {'original': 'sadness'},
        'surprise': {'original': 'surprise'}
    }
    emotion_mapping = {
        'anger': 'anger',
        'disgust': 'disgust',
        'fear': 'fear',
        'joy': 'happiness',
        'neutrality': 'neutral',
        'no_agreement': 'no_agreement',
        'sadness': 'sadness',
        'surprise': 'surprise'
    }

    user_df = pd.read_csv(user_path)
    user_df.set_index('username', inplace=True)
    samples_df = pd.read_csv(samples_path, 
        converters={'file_name': lambda x: os.path.join(audio_dir_name, x)})
    evaluations_df = pd.read_csv(evaluations_path,
        converters={'file_name': lambda x: os.path.join(audio_dir_name, x)})
    samples_df['emotion_expressed'] = samples_df['emotion_expressed'].map(emotion_mapping)
    evaluations_df['emotion_recognized'] = evaluations_df['emotion_recognized'].map(emotion_mapping)

    # -------- Add schemes --------
    speakers = samples_df['actor'].unique()
    speaker_info = {}
    for speaker in speakers:
        speaker_row = user_df.loc[speaker]
        speaker_info[speaker] = {
            'gender': speaker_row['gender'], 'age': int(speaker_row['age']),
            'mother_tongue': speaker_row['mother_tongue']
        }

    db.schemes['speaker'] = audformat.Scheme(
        labels=speaker_info
    )

    db.schemes['emotion'] = audformat.Scheme(
        labels=emotion_labels
    )
    db.schemes['emotion.agreement'] = audformat.Scheme(
        dtype='float', minimum=0, maximum=1,
    )

    db.schemes['votes'] = audformat.Scheme(
        dtype='int', minimum=0,
    )

    sentences = samples_df['sentence'].unique()
    transcriptid2sentence = {
        f's{i}': sentence for i, sentence in enumerate(sentences)
    }
    sentence2transcriptid = {
        v:k for k,v in transcriptid2sentence.items()
    }
    db.schemes['transcription'] = audformat.Scheme(
        labels = transcriptid2sentence
    )
    samples_df['transcription'] = samples_df['sentence'].map(sentence2transcriptid)

    # -------- Gold standard of emotion votes --------

    emotion_gold_standard = evaluations_df.groupby('file_name')['emotion_recognized'].agg(
        gold_standard
    )
    emotion_agreement = evaluations_df.groupby('file_name')['emotion_recognized'].agg(
        categorical_voter_percentage
    )
    emotion_agreement.name = 'emotion.agreement'

    samples_df = samples_df.merge(emotion_gold_standard, on='file_name')
    samples_df = samples_df.merge(emotion_agreement, on='file_name')
    samples_df.index = audformat.filewise_index(samples_df['file_name'])

    # -------- Rater values --------
    votes_df = evaluations_df.groupby(['file_name', 'emotion_recognized']).size().unstack(fill_value=0)
    votes_df.index = audformat.filewise_index(votes_df.reset_index()['file_name'])
    votes_df.reindex(samples_df.index)  # use same order for index as for other tables

    # -------- Files table with speaker information --------

    db['files'] = audformat.Table(
        index=samples_df.index
    )

    db['files']['speaker'] = audformat.Column(
        scheme_id='speaker'
    )
    db['files']['speaker'].set(samples_df['actor'])
    db['files']['transcription'] = audformat.Column(
        scheme_id='transcription'
    )
    db['files']['transcription'].set(samples_df['transcription'])


    # -------- Load speaker splits as created in the notebook --------
    speaker_split_path = os.path.join(current_dir, 'speaker_splits.json')
    with open(speaker_split_path, 'r') as fp:
        speaker_info = json.load(fp)


    for split, split_speakers in speaker_info.items():
        # Create db splits
        db.splits[split] = audformat.Split(type=split, description=f'Unofficial speaker independent {split} split')

        split_df = samples_df[samples_df['actor'].isin(split_speakers)]
        # Emotion categories the actors tried to portray
        db[f'emotion.categories.desired.{split}'] = audformat.Table(
            index=split_df.index, split_id=split, description='The categorical emotions the actors were asked '
                                                              f' to portray (unofficial speaker independent '
                                                              f'{split} split).'
        )
        db[f'emotion.categories.desired.{split}']['emotion'] = audformat.Column(
            scheme_id='emotion',
        )
        db[f'emotion.categories.desired.{split}']['emotion'].set(split_df['emotion_expressed'])


        # Emotion categories the raters perceived

        db[f'emotion.categories.{split}.votes'] = audformat.Table(
            index=split_df.index, description='The votes for each emotion (unofficial speaker '
                                              f'independent {split} split)'
        )
        for emotion in emotion_labels:
            if emotion == 'no_agreement':
                continue
            db[f'emotion.categories.{split}.votes'][emotion] = audformat.Column(
                scheme_id='votes', description=f'The number of times raters voted for {emotion}.'
            )
            db[f'emotion.categories.{split}.votes'][emotion].set(
                votes_df.loc[split_df.index, emotion]
            )

        # Gold standard of emotion categories (by raters)

        db[f'emotion.categories.{split}.gold_standard'] = audformat.Table(
            index=split_df.index, description=f'The gold standard table of the unofficial speaker independent '
                                              f'{split} split for emotion categories.'
        )

        db[f'emotion.categories.{split}.gold_standard']['emotion'] = audformat.Column(
            scheme_id='emotion', description=f'The most commonly voted for categorical emotion, or no_agreement '
                                             f'when there is more than one.'
        )
        db[f'emotion.categories.{split}.gold_standard']['emotion'].set(
            split_df['emotion_recognized']
        )
        db[f'emotion.categories.{split}.gold_standard']['emotion.agreement'] = audformat.Column(
            scheme_id='emotion.agreement', description=f'The proportion of raters that voted for the most commonly '
                                                       f'selected emotion.'
        )
        db[f'emotion.categories.{split}.gold_standard']['emotion.agreement'].set(
            split_df['emotion.agreement']
        )

        if not os.path.exists(os.path.join(build_dir, audio_dir_name)):
            shutil.copytree(
                audio_path,
                os.path.join(current_dir, build_dir, audio_dir_name)
            )
        db.save(build_dir)

def gold_standard(votes):
    mode = pd.Series.mode(votes)
    if len(mode)==1:
        return mode
    else:
        return 'no_agreement'

def categorical_voter_percentage(votes):
    mode = pd.Series.mode(votes)
    if len(mode) == 0:
        return np.NaN
    mode = mode[0]
    max_n_votes = sum(votes == mode)
    total_votes = len(votes)
    return max_n_votes / total_votes

if __name__ == '__main__':
    main()
