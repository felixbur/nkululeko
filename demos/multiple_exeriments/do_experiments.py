import os

classifiers = [
    {'--model': 'xgb'},
    {'--model': 'svm'},
]

features = [
    {'--feat': 'os'},
    {'--feat': 'os', 
    '--set': 'ComParE_2016',
    },
    {'--feat': 'mld'},
    {'--feat': 'mld',
    '--with_os': 'True',
    },
    {'--feat': 'xbow'},
    {'--feat': 'xbow',
    '--with_os': 'True',
    },
    {'--feat': 'trill'},
    {'--feat': 'wav2vec'},
]


for c in classifiers:
    for f in features:
        cmd = f'python parse_nkulu.py '
        for item in c:
            cmd += f'{item} {c[item]} '
        for item in f:
            cmd += f'{item} {f[item]} '
        print(cmd)
        os.system(cmd)