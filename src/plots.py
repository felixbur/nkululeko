# plots.py
import pandas as pd
import matplotlib.pyplot as plt

    

def describe_df(df, target, save_path):
    """Make a stacked barplot of samples and speakers per sex and target values. speaker, gender and target columns must be present"""
    print('# samples: {}, # speakers: {}'.format(df.shape[0], df.speaker.nunique()))
    fig, axes = plt.subplots(nrows=1, ncols=2)
    df.groupby('emotion')['gender'].value_counts().unstack().plot(kind='bar', stacked=True, ax=axes[0], title='samples')
    df.groupby(target)['speaker'].nunique().plot(kind='bar', ax=axes[1], title='speakers')
    plt.tight_layout()
    plt.savefig(save_path)
    fig.clear()
    plt.close(fig)