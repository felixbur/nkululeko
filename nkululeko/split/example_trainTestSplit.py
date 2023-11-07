"""
Code copyright by Uwe Reichel
"""

import pandas as pd
import audb
from split_utils import optimize_traintest_split

# define testset on emodb, that is:
#   - speaker disjunct
#   - optimally stratified on emotion
#   - optimally stratified on gender
#   - optimally stratified on transcriptions
#   - that contains 10% of the speakers
#   - and approximately 10% of the files

# data
db = audb.load(
    "emodb", version="1.3.0", format="wav", sampling_rate=16000, mixdown=True
)
df_emotion = db["emotion"].get()
df_files = db["files"].get()
df_speaker = db["speaker"].get()
df = pd.concat([df_emotion, df_files], axis=1, join="inner")


def spk2gender(x):
    if x in [8, 9, 13, 14, 16]:
        return "female"
    return "male"


df["gender"] = df["speaker"].map(spk2gender)

# seed, test proportion, number of different splits
seed = 42
test_size = 0.2
k = 30

# targets
emotion = df["emotion"].to_numpy()

# on which variable to split

speaker = df["speaker"].to_numpy()

# on which variables (targets, groupings) to stratify
stratif_vars = {
    "emotion": emotion,
    "gender": df["gender"].to_numpy(),
    "transcription": df["transcription"].to_numpy(),
}

# weights for all stratify_on variables and
# and for test proportion match. Give target
# variable EMOTION more weight than groupings.
weight = {"emotion": 2, "gender": 1, "transcription": 1, "size_diff": 1}

# find optimal test indices TEST_I in DF
# info: dict with goodness of split information
train_i, test_i, info = optimize_traintest_split(
    X=df,
    y=emotion,
    split_on=speaker,
    stratify_on=stratif_vars,
    weight=weight,
    test_size=test_size,
    k=k,
    seed=seed,
)

print("test split of DF:")
print(df.iloc[test_i])
print("test split of target variable:")
print(emotion[test_i])
print("goodness of split:")
print(info)
