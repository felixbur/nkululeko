import os

src_path = 'demos/multiple_exeriments/'

classifiers = [
    {'--model': 'mlp',
    '--layers': '\"{\'l1\':64,\'l2\':16}\"'},
    {'--model': 'mlp',
    '--layers': '\"{\'l1\':64,\'l2\':16}\"',
    '--learning_rate': '.1',},
    {'--model': 'mlp',
    '--layers': '\"{\'l1\':64,\'l2\':16}\"',
    '--learning_rate': '.0001',
    '--drop': '.3',
    },
    {'--model': 'xgb',
    '--epochs':1},
    {'--model': 'svm',
    '--epochs':1},
]

features = [
    {'--feat': 'os'},
    # {'--feat': 'os', 
    # '--set': 'ComParE_2016',
    # },
    # {'--feat': 'mld'},
    # {'--feat': 'mld',
    # '--with_os': 'True',
    # },
    # {'--feat': 'xbow'},
    # {'--feat': 'xbow',
    # '--with_os': 'True',
    # },
    # {'--feat': 'trill'},
    # {'--feat': 'wav2vec'},
]


for c in classifiers:
    for f in features:
        cmd = f'python {src_path}parse_nkulu.py '
        for item in c:
            cmd += f'{item} {c[item]} '
        for item in f:
            cmd += f'{item} {f[item]} '
        print(cmd)
        os.system(cmd)