# PVQD database

From [the website](https://data.mendeley.com/datasets/9dz247gnyb/1)

*This database was created through generous funding from The Voice Foundation's Advancing Scientific Voice Research Grant and contains voice samples which have been rated by experienced voice professionals (at least 3 different raters with a minimum of 3 years’ clinical experience) in order to provide educators with standardized materials to better train pre-service clinical voice professionals. It contains 296 audio files consisting of the sustained /a/ and /i/ vowels and the sentences from the Consensus Auditory-Perceptual Evaluation of Voice (CAPE-V; Kempster, 2007). All recordings were made in a quiet clinical environment using a head-mounted condenser microphone at a 6-centimeter distance from the corner of the mouth and the Computerized Speech Lab (CSL) using 16-bit encryption and a sampling rate of 48k. Audio recordings have been edited as best as possible to remove all clinician instructions. However, please listen to and look at each file carefully just in case there was simultaneous clinician-client talk.*

*Listeners rated approximately 50 files each and each file was rated twice for reliability measurement (for a total of approximately 100 ratings per rater). Raters used a computer to listen to the samples and rate voice quality via a web-based system that included custom-made electronic scales for the CAPE-V (Kempster, 2007) and the GRBAS (Hirano, 1981) using Qualtrics survey software. Listeners rated each file on a 100-point visual analogue scale (VAS) to mimic the paper-based CAPE-V protocol. Please note that severity markers (mild, moderate, severe) were not included on the 100-point VAS to avoid influencing the concurrent rating using the GRBAS scale. Raters were urged to rate the samples over several days to avoid fatigue. Further description of methods is located in the folders below.*
*Questions about the database can be directed to Patrick R. Walden, Ph.D., CCC-SLP at waldenp@stjohns.edu.*

## Create
* Download the zip repo from the website
* unzip it in a folder called *download_data*
* run ```python process_database.py```


## Test
* run ```python -m nkululeko.nkululeko --config exp.ini```