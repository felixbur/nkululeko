import os

src_path = './demos/multiple_exeriments/'


features = [
    {'--feat': '[\\\'os\\\']'},
    {'--feat': '[\\\'wav2vec\\\']'},
    {'--feat': '[\\\'trill\\\']'},
    {'--feat': '[\\\'praat\\\']'},
    {'--feat': '[\\\'os\\\',\\\'praat\\\']'},
    {'--feat': '[\\\'wav2vec\\\',\\\'praat\\\']'},
    {'--feat': '[\\\'trill\\\',\\\'praat\\\']'},
]


for f in features:
    cmd = f'python {src_path}parse_nkulu.py --config "./tests/exp_multiple_cross.ini" --model mlp --learning_rate 0.001 --drop 0.2 '
    for item in f:
        cmd += f'{item} {f[item]} '
    print(cmd)
    os.system(cmd)