import audformat
import pandas as pd
import nkululeko.glob_conf as glob_conf

def limit_speakers(df, max=20):
    """ limit number of samples per speaker
        the samples are selected randomly          
    """
    df_ret = pd.DataFrame()
    for s in df.speaker.unique():
        s_df = df[df['speaker'].eq(s)]
        if s_df.shape[0] < max:
            df_ret = df_ret.append(s_df)
        else:
            df_ret = df_ret.append(s_df.sample(max))
    return df_ret

def filter_min_dur(df, min_dur):
    """remove all samples less than min_dur duration
    """
    df_ret = df.copy()
    if not isinstance(df.index, pd.MultiIndex):
        glob_conf.util.debug('converting file index to multi index, this might take a while...')
        df_ret.index = audformat.utils.to_segmented_index(df.index, allow_nat=False)    
    for i in df_ret.index:
        start = i[1]
        end = i[2]
        dur = (end - start).total_seconds()
        if dur < float(min_dur):
            df_ret = df_ret.drop(i, axis=0)
    df_ret.is_labeled = df.is_labeled
    df_ret.got_gender = df.got_gender
    df_ret.got_speaker = df.got_speaker
    return df_ret

def filter_max_dur(df, max_dur):
    """remove all samples less than min_dur duration
    """
    df_ret = df.copy()
    if not isinstance(df.index, pd.MultiIndex):
        glob_conf.util.debug('converting file index to multi index, this might take a while...')
        df_ret.index = audformat.utils.to_segmented_index(df.index, allow_nat=False)    
    for i in df_ret.index:
        start = i[1]
        end = i[2]
        dur = (end - start).total_seconds()
        if dur > float(max_dur):
            df_ret = df_ret.drop(i, axis=0)
    df_ret.is_labeled = df.is_labeled
    df_ret.got_gender = df.got_gender
    df_ret.got_speaker = df.got_speaker
    return df_ret
