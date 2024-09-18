import os

classifiers = [
    {"--model": "mlp", "--layers": "\"{'l1':64,'l2':16}\"", "--epochs": 100},
    {
        "--model": "mlp",
        "--layers": "\"{'l1':128,'l2':64,'l3':16}\"",
        "--learning_rate": ".01",
        "--drop": ".3",
        "--epochs": 100,
    },
    {"--model": "xgb", "--epochs": 1},
    {"--model": "svm", "--epochs": 1},
]

features = [
    {"--feat": "os"},
    # {'--feat': 'os',
    # '--set': 'ComParE_2016',
    # },
    {"--feat": "praat"},
]


for c in classifiers:
    for f in features:
        cmd = "python -m nkululeko.nkuluflag --config meta/demos/multiple_exeriments/exp.ini "
        for item in c:
            cmd += f"{item} {c[item]} "
        for item in f:
            cmd += f"{item} {f[item]} "
        print(cmd)
        os.system(cmd)
        # print(f"results: {result}, {last_epoch}")
