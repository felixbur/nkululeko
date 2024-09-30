import pandas as pd


def process_database(output_path):
    # Read the list files
    train_normal = pd.read_csv('train_normal.lst', header=None)
    train_pathology = pd.read_csv('train_pathology.lst', header=None)
    dev_normal = pd.read_csv('develop_normal.lst', header=None)
    dev_pathology = pd.read_csv('develop_pathology.lst', header=None)
    test_normal = pd.read_csv('test_normal.lst', header=None)
    test_pathology = pd.read_csv('test_pathology.lst', header=None)

    # Process and save train, dev, and test sets
    for name, normal, pathology in [
        ('train', train_normal, train_pathology),
        ('dev', dev_normal, dev_pathology),
        ('test', test_normal, test_pathology)
    ]:
        audio_types = [
            'phrase', 
            'a_n', 
            'a_l', 
            'a_h',
            'a_lhl',
            'i_n',
            'i_l'
            'i_h',
            'i_lhl',
            'u_n',
            'u_l',
            'u_h',
            'u_lhl',
            'all'
            'aiu'
            'a',
            'i',
            'u',
            'aiu_n',
            'aiu_nlh'
            ]
        for audio_type in audio_types:
            df = pd.concat([
                normal[0].apply(lambda x: f'{x}-{audio_type}.wav').to_frame('file').assign(label='n'),
                pathology[0].apply(lambda x: f'{x}-{audio_type}.wav').to_frame('file').assign(label='p')
            ])[['file', 'label']]
            df.to_csv(f'{output_path}/svd_{audio_type}_{name}.csv', index=False)
        print(f"Number of {name} samples: {len(df)}")
    print("Done!")

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=".", help="Path to store processed data")
    args = parser.parse_args()
    process_database(args.out_dir)
