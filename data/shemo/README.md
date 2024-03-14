# Nkululeko pre-processing dataset for ShEMO (Public)  

This paper introduces a large-scale, validated database for Persian called Sharif Emotional Speech Database (ShEMO). The database includes 3000 semi-natural utterances, equivalent to 3 hours and 25 minutes of speech data extracted from online radio plays. The ShEMO covers speech samples of 87 native-Persian speakers for five basic emotions including anger, fear, happiness, sadness and surprise, as well as neutral state. Twelve annotators label the underlying emotional state of utterances and majority voting is used to decide on the final labels. According to the kappa measure, the inter-annotator agreement is 64% which is interpreted as "substantial agreement". We also present benchmark results based on common classification methods in speech emotion detection task. According to the experiments, support vector machine achieves the best results for both gender-independent (58.2%) and gender-dependent models (female=59.4%, male=57.6%). The ShEMO is available for academic purposes free of charge to provide a baseline for further research on Persian emotional speech. 

**The ZIP files in the following Github link is broken, plese follow the bash commands below [2]: https://github.com/mansourehk/ShEMO** 


```bash
$ wget -O female.zip "https://www.dropbox.com/s/4t6mep8mo4yf81f/female.zip?dl=0"
$ wget -O male.zip "https://www.dropbox.com/s/xfi3hi927yxixa9/male.zip?dl=0"
$ unzip female.zip
$ unzip male.zip
$ python3 process_database.py
$ python3 -m nkululeko.resample --config data/shemo/exp.ini
$ python3 -m nkululeko.nkululeko --config data/shemo/exp.ini
```


References:  
[1] Mohamad Nezami, O., Jamshid Lou, P., & Karami, M. (2019). ShEMO: a large-scale validated database for Persian speech emotion detection. Language Resources and Evaluation, 53(1), 1–16. https://doi.org/10.1007/s10579-018-9427-x
[2] https://github.com/pariajm/sharif-emotional-speech-dataset