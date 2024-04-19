# Hello World [CSV]

In the previous tutorial, we learned how to use Nkululeko with audformat dataset.
Since most dataset are not in audformat, we will learn how to use Nkululeko with pre-processed CSV dataset.

## Dataset Location

The dataset is assumed to be located in the `data` directory under `Nkululeko` root directory. The best practice is to store dataset in `/data/` or `/home/$USER/data/` directory and then make a symbolic link to each dataset in the Nkululeko `data` directory.
Here the example of downloading dataset into its location, doing pre-processing and running the experiment. The main idea of the pre-processing is to convert the dataset into the format that Nkululeko can understand. Usually, the pre-processing is done by running the `process_database.py` script. You can learn more about the pre-processing in each dataset directory (`/nkululeko/data`).

Let's start with the ravdess directory.

This `ravdess` folder is to import the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
database to nkululeko.

We used the version downloadable from [Zenodo](https://zenodo.org/record/1188976)

Download and unzip the file Audio_Speech_Actors_01-24.zip

```bash
wget https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip
unzip Audio_Speech_Actors_01-24.zip
```

Run the file

```bash
python3 process_database.py
```

Change to Nkululeko parent directory,

```bash
cd ../..
```

then, as a test, you might do

```bash
python3 -m nkululeko.nkululeko --config data/ravdess/exp_ravdess_os_xgb.ini 
```

Check the results in the results folder under Nkululeko parent directory.

Just simple as that. Check your results and play with some parameters. If facing any problem, please open an issue in [our github](https://github.com/felixbur/nkululeko/).
