# Demo module

Nkululeko has `demo` module to either test the built model with an audio file,
a list of audio files, or even live streaming from your microphone!

Let's recap the concept of train/dev/test splits:  

- train is used to train a supervised model  
- dev is a set to evaluate this model, i.e. know when it is a good model (that doesn't overfit)  
- test is a set to be used ONLY once or hidden: for the real use of the model. If you would use the test as a dev set, you can't be sure if you're not overfitting again (because you used the dev set to adjust the meta parameters of your model).  

Using machine learning (like SVM or XGB), usually we only need train and test.
Using neural networks or deep learning (like MLP or CNN), we need train, dev, and test.

In this tutorial, we split the use case of `demo` module into two:  
(1) predicting audio file or
live streaming from microphone and  
(2) predicting a list of audio files.  

Let's start case by case.

## Needed Configuration For Demo Module

In order to use a model, of course you do need to have it trained and saved before. So you need a run with the nkululeko module before. Below is minimal addition to the [INI FILE](ini_file.md) needed to run the demo module.

```ini
[MODEL]
save = True
```

## Predicting an audio file or live streaming from microphone

To predict an audio file or live streaming from microphone, you can use the following command:

```bash
python3 -m nkululeko.demo --config data/ravdess/exp_ravdess_os_xgb.ini
```

If no argument optional argument is given, the demo will try to getting signal from microphone. If `--file` argument is given, the `demo` will predict the audio file. An example is given below,

```bash
(.env) bagus@pc-omen:nkululeko (docs)$ python3 -m nkululeko.demo --config data/ravdess/exp1.ini --file data/ravdess/Actor_01/03-01-01-01-01-01-01.wav
DEBUG demo: running exp_ravdess_os3 from config data/ravdess/exp1.ini, nkululeko version 0.81.4
DEBUG model: value for C_val not found, using default: 0.001
DEBUG model: value for kernel not found, using default: rbf
predicting file: data/ravdess/Actor_01/03-01-01-01-01-01-01.wav, len: 52853 bytes, sampling rate: 16000
{'angry': '0.314', 'happy': '0.312', 'neutral': '0.042', 'sad': '0.332', 'predicted': 'sad'}
DONE
```

## Predicting a list of audio files

To predict a list of audio files, you can use `--list` argument. An example is given below. Notice, that you need to specify the relative path of audio file in that list if it is not relative to the current directory. Additonally, you can add`--outfile` argument to save the result to a file.

```bash
(.env) python3 -m nkululeko.demo --config data/ravdess/exp1.ini --list data/ravdess/ravdess_test.csv --folder data/ravdess/ --outfile /tmp/ravdess_test_predict.csv
DEBUG demo: running exp_ravdess_os3 from config data/ravdess/exp1.ini, nkululeko version 0.81.4
DEBUG model: value for C_val not found, using default: 0.001
DEBUG model: value for kernel not found, using default: rbf
predicting file data/ravdess/./Actor_21/03-01-07-01-01-01-21.wav
{'angry': '0.314', 'happy': '0.315', 'neutral': '0.038', 'sad': '0.332', 'predicted': 'sad'}
...
predicting file data/ravdess/./Actor_24/03-01-08-01-01-02-24.wav
{'angry': '0.314', 'happy': '0.314', 'neutral': '0.040', 'sad': '0.332', 'predicted': 'sad'}
predicting file data/ravdess/./Actor_24/03-01-03-01-01-02-24.wav
{'angry': '0.314', 'happy': '0.311', 'neutral': '0.044', 'sad': '0.331', 'predicted': 'sad'}
predicting file data/ravdess/./Actor_24/03-01-08-02-01-02-24.wav
DONE
```

The example of CSV file for prediction is given below. You can submit it for a such challange.

```
file,angry,happy,neutral,sad,predicted
./Actor_21/03-01-07-01-01-01-21.wav,0.314,0.315,0.038,0.332,sad
./Actor_21/03-01-06-01-02-02-21.wav,0.314,0.313,0.041,0.332,sad
./Actor_21/03-01-06-02-01-02-21.wav,0.314,0.316,0.037,0.332,sad
```

List of arguments for `demo` module is given below.

  * `--list` (optional) list of input files
  * `--file` (optional) name of input file
  * `--folder` (optional) parent folder for input files
  * `--outfile` (optional) name of CSV file for output