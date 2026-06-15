# Adopted from Monica Gonzales Manchorro

import os
import shutil
import re

import pandas as pd
import numpy as np

import audformat
import audeer

import preprocess

def extract_prefix(filename):
    match = re.match(r"^([A-Za-z]+[0-9]+)", filename)
    if match:
        return match.group(1)
    else:
        return None


def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    db_root = os.path.join(root_dir, "db")
    data_path = os.path.join(root_dir, "download_data")
    excel_path = os.path.join("Ratings Spreadsheets")
    audio_path = os.path.join("Audio Files")

    description = "The Perceptual Voice Qualities Database (PVQD) contains voice samples which have been rated by experienced voice professionals (at least 3 different raters with a minimum of 2 years’ clinical experience) in order to provide educators with standardized materials to better train pre-service clinical voice professionals. It contains 296 audio files consisting of the sustained /a/ and /i/ vowels and the sentences from the Consensus Auditory-Perceptual Evaluation of Voice (CAPE-V). All recordings were made in a quiet clinical environment using a head-mounted condenser microphone at a 6-centimeter distance from the corner of the mouth and the Computerized Speech Lab (CSL) using 16-bit quantization and a sampling rate of 44.1K.. Audio recordings have been edited as best as possible to remove all clinician instructions."

    EN = audformat.utils.map_language("en")
    db = audformat.Database(
        name="pvqd-dysphonia",
        usage=audformat.define.Usage.RESEARCH,
        description=description,
        languages=[EN],
        source = "https://data.mendeley.com/datasets/9dz247gnyb/1"
    )


    build_dir = audeer.mkdir(db_root)


    df = pd.read_excel(os.path.join(data_path, excel_path, "Demographics.xlsx"))
    df.rename(columns={"Participant ID ": "participant_code"}, inplace=True)
    df.columns = df.columns.str.lower()
    list_file_names = audeer.list_file_names(os.path.join(data_path, audio_path), basenames=True)
    df_audios = pd.DataFrame({"file": list_file_names})

    # identify participant_code
    spk_id = []
    for file in df_audios["file"]:
        speaker_id = extract_prefix(file)
        spk_id.append(speaker_id)
    df_audios["participant_code"] = spk_id
    
    # merge df_audios with df
    df_metadata = df_audios.merge(df, on="participant_code")

    # add medical scores: GRBAS score
    df_grbas = pd.read_excel(os.path.join(data_path, excel_path, "grbas_grade_only.xlsx"))
    df_grbas.rename(
        columns={
            "File": "participant_code",
            "Average": "grbas_score",
            "Category": "grbas_category",
        },
        inplace=True,
    )
    df_grbas.columns = df_grbas.columns.str.lower()
   
    # merge medical scores with metadata
    df_metadata = df_metadata.merge(df_grbas, on="participant_code")

    df_metadata["gender"] = df_metadata["gender"].replace({"F": "female", "M": "male"})
    df_metadata['grbas_category'] = df_metadata['grbas_category'].str.lower()
    df_metadata["file"] = df_metadata["file"].apply(
        lambda file_name: os.path.join("data/", audio_path, file_name)
    )

    df_metadata.set_index(["file"], inplace=True)

    # define schemes db
    db.media["microphone"] = audformat.Media(
        audformat.define.MediaType.AUDIO,
        format="wav",
    )

    db.schemes["gender"] = audformat.Scheme(
        audformat.define.DataType.STRING, description="Gender of the speaker"
    )

    db.schemes["age"] = audformat.Scheme(
        audformat.define.DataType.INTEGER, description="Age of the speaker"
    )

    db.schemes["speaker"] = audformat.Scheme(
        audformat.define.DataType.STRING, description="ID of the speaker"
    )

    db.schemes["grbas_score"] = audformat.Scheme(
        audformat.define.DataType.FLOAT,
        description="Grade or overall severity of dysphonia. The rater examines the speaker's voice on a 4-point scale: 1, without disorder; 2, mild disorder; 3, moderated disorder; 4, severe disorder",
    )

    db.schemes["grbas_category"] = audformat.Scheme(
        audformat.define.DataType.STRING,
        labels=["mild", "moderate","severe", "normal"],
        description="Category for the grade or overall severity of dysphonia. This category is the results of the GRBAS score",
    )

    table_id = "files"
    db[table_id] = audformat.Table(index=df_metadata.index, media_id="microphone")

    # assign schemes to columns
    
    db[table_id]["gender"] = audformat.Column(scheme_id="gender")
    db[table_id]["gender"].set(df_metadata["gender"])
    
    db[table_id]["speaker"] = audformat.Column(scheme_id="speaker")
    db[table_id]["speaker"].set(df_metadata["participant_code"])
    
    db[table_id]["age"] = audformat.Column(scheme_id="age")
    db[table_id]["age"].set(df_metadata["age"])
    
    db[table_id]["grbas_category"] = audformat.Column(scheme_id="grbas_category")
    db[table_id]["grbas_category"].set(df_metadata["grbas_category"])
    
    db[table_id]["grbas_score"] = audformat.Column(scheme_id="grbas_score")
    db[table_id]["grbas_score"].set(df_metadata["grbas_score"])

    """
        -a new column for files tables, containing a categorization of the severity of aphasia
    """

    df_aesthenia = pd.read_excel(os.path.join(data_path, excel_path, "grbas_asthenia_only.xlsx"))
    df_aesthenia.rename(
        columns={
            "File": "speaker",
            "Average Formula": "grbas_aesthenia_score",
            "Category Values": "grbas_aesthenia_category",
        },
        inplace=True,
    )
    df_aesthenia = df_aesthenia[
        ["speaker", "grbas_aesthenia_score", "grbas_aesthenia_category"]
    ]

    df_roughness = pd.read_excel(os.path.join(data_path, excel_path, "grbas_roughness_only.xlsx"))
    df_roughness.rename(
        columns={
            "File": "speaker",
            "Average Values": "grbas_roughness_score",
            "Category Values": "grbas_roughness_category",
        },
        inplace=True,
    )

    df_roughness = df_roughness[
        ["speaker", "grbas_roughness_score", "grbas_roughness_category"]
    ]

    df_breathiness = pd.read_excel(os.path.join(data_path, excel_path, "grbas_breathiness_only.xlsx"))
    
    df_breathiness.rename(
        columns={
            "File": "speaker",
            "Average Value": "grbas_breathiness_score",
            "Category Value": "grbas_breathiness_category",
        },
        inplace=True,
    )
    df_breathiness = df_breathiness[
        ["speaker", "grbas_breathiness_score", "grbas_breathiness_category"]
    ]

    df_strain = pd.read_excel(os.path.join(data_path, excel_path, "grbas_strain_only.xlsx"))
    df_strain.rename(
        columns={
            "File": "speaker",
            "Average Values": "grbas_strain_score",
            "Category Value": "grbas_strain_category",
        },
        inplace=True,
    )
    df_strain = df_strain[["speaker", "grbas_strain_score", "grbas_strain_category"]]

    # Merge all DataFrames on 'speaker'
    df_metadata = (
        df_aesthenia.merge(df_roughness, on="speaker", how="outer")
        .merge(df_breathiness, on="speaker", how="outer")
        .merge(df_strain, on="speaker", how="outer")
    )
    # Convert all columns ending with '_category' to lowercase
    for col in df_metadata.columns:
        if col.endswith("_category"):
            df_metadata[col] = df_metadata[col].str.lower()

    files = db["files"].get()

    df = files.merge(df_metadata, on="speaker")
    # Normalize gender column to be always 'female' or 'male'
    if "gender" in df.columns:
        df["gender"] = (
            df["gender"]
            .str.lower()
            .map({"f": "female", "female": "female", "m": "male", "male": "male"})
        )

    db.schemes["grbas_aesthenia_score"] = audformat.Scheme(
        audformat.define.DataType.FLOAT,
        description="Aesthenia score of the GRBAS scale",
    )
    db.schemes["grbas_aesthenia_category"] = audformat.Scheme(
        audformat.define.DataType.STRING,
        labels=["mild", "moderate", "severe", "normal"],
        description="Aesthenia category of the GRBAS scale. Each component is rated on an integer four point scale, in which 0 is normal, 1 slight, 2 moderate, and 3 severe",
    )

    db.schemes["grbas_roughness_score"] = audformat.Scheme(
        audformat.define.DataType.FLOAT,
        description="Roughness score of the GRBAS scale",
    )
    db.schemes["grbas_roughness_category"] = audformat.Scheme(
        audformat.define.DataType.STRING,
        labels=["mild", "moderate", "severe", "normal"],
        description="Roughness category of the GRBAS scale. Each component is rated on an integer four point scale, in which 0 is normal, 1 slight, 2 moderate, and 3 severe",
    )
    db.schemes["grbas_breathiness_score"] = audformat.Scheme(
        audformat.define.DataType.FLOAT,
        description="Breathiness score of the GRBAS scale",
    )
    db.schemes["grbas_breathiness_category"] = audformat.Scheme(
        audformat.define.DataType.STRING,
        labels=["mild", "moderate", "severe", "normal"],
        description="Breathiness category of the GRBAS scale. Each component is rated on an integer four point scale, in which 0 is normal, 1 slight, 2 moderate, and 3 severe",
    )
    db.schemes["grbas_strain_score"] = audformat.Scheme(
        audformat.define.DataType.FLOAT,
        description="Strain score of the GRBAS scale",
    )
    db.schemes["grbas_strain_category"] = audformat.Scheme(
        audformat.define.DataType.STRING,
        labels=["mild", "moderate", "severe", "normal"],
        description="Strain category of the GRBAS scale. Each component is rated on an integer four point scale, in which 0 is normal, 1 slight, 2 moderate, and 3 severe",
    )

    table_id = "files"
    db[table_id]["grbas_aesthenia_score"] = audformat.Column(
        scheme_id="grbas_aesthenia_score"
    )
    db[table_id]["grbas_aesthenia_score"].set(df["grbas_aesthenia_score"])

    db[table_id]["grbas_aesthenia_category"] = audformat.Column(
        scheme_id="grbas_aesthenia_category"
    )
    db[table_id]["grbas_aesthenia_category"].set(df["grbas_aesthenia_category"])

    db[table_id]["grbas_roughness_score"] = audformat.Column(
        scheme_id="grbas_roughness_score"
    )
    db[table_id]["grbas_roughness_score"].set(df["grbas_roughness_score"])

    db[table_id]["grbas_roughness_category"] = audformat.Column(
        scheme_id="grbas_roughness_category"
    )
    db[table_id]["grbas_roughness_category"].set(df["grbas_roughness_category"])

    db[table_id]["grbas_breathiness_score"] = audformat.Column(
        scheme_id="grbas_breathiness_score"
    )
    db[table_id]["grbas_breathiness_score"].set(df["grbas_breathiness_score"])

    db[table_id]["grbas_breathiness_category"] = audformat.Column(
        scheme_id="grbas_breathiness_category"
    )
    db[table_id]["grbas_breathiness_category"].set(df["grbas_breathiness_category"])

    db[table_id]["grbas_strain_score"] = audformat.Column(
        scheme_id="grbas_strain_score"
    )
    db[table_id]["grbas_strain_score"].set(df["grbas_strain_score"])
    
    db[table_id]["grbas_strain_category"] = audformat.Column(
        scheme_id="grbas_strain_category"
    )
    db[table_id]["grbas_strain_category"].set(df["grbas_strain_category"])

    db[table_id]["gender"] = audformat.Column(scheme_id="gender")
    db[table_id]["gender"].set(df["gender"])

    # copy audio files in data to build dir
    shutil.copytree(
        data_path,
        os.path.join(build_dir, "data/"),
        dirs_exist_ok=True,
    )


    """adds spectral instability and a column called speech task
    creates splits for training and testing"""

    files = db["files"].get()

    files.index = build_dir + "/" + files.index.astype(str)
    df_files = preprocess.vad_segmentation(files, strip_prefix=build_dir)


    df = preprocess.create_train_test_split(df_files)

    input(df)

    db.schemes["speech_task"] = audformat.Scheme(
        audformat.define.DataType.STRING,
        description="type of speech task corresponding to the segment",
        labels=["read_speech", "sustained_utterance"],
    )

    table_id = "segments"

    splits = ["train", "test"]
    for split in splits:
        table_id = f"segments.{split}"

        df_split = df[df["split"] == split].reset_index(drop=True)

        df_split.set_index(["file", "start", "end"], inplace=True)
        
        db[table_id] = audformat.Table(index=df_split.index, media_id="microphone")

        db[table_id]["speech_task"] = audformat.Column(scheme_id="speech_task")
        db[table_id]["speech_task"].set(df_split["speech_task"])
        
        db[table_id]["speaker"] = audformat.Column(scheme_id="speaker")
        db[table_id]["speaker"].set(df_split["speaker"])

        db[table_id]["age"] = audformat.Column(scheme_id="age")
        db[table_id]["age"].set(df_split["age"])

        db[table_id]["grbas_category"] = audformat.Column(scheme_id="grbas_category")
        db[table_id]["grbas_category"].set(df_split["grbas_category"])

        db[table_id]["grbas_score"] = audformat.Column(scheme_id="grbas_score")
        db[table_id]["grbas_score"].set(df_split["grbas_score"])

        db[table_id]["grbas_aesthenia_score"] = audformat.Column(
        scheme_id="grbas_aesthenia_score"
        )
        db[table_id]["grbas_aesthenia_score"].set(df_split["grbas_aesthenia_score"])

        db[table_id]["grbas_aesthenia_category"] = audformat.Column(
            scheme_id="grbas_aesthenia_category"
        )
        db[table_id]["grbas_aesthenia_category"].set(df_split["grbas_aesthenia_category"])

        db[table_id]["grbas_roughness_score"] = audformat.Column(
            scheme_id="grbas_roughness_score"
        )
        db[table_id]["grbas_roughness_score"].set(df_split["grbas_roughness_score"])

        db[table_id]["grbas_roughness_category"] = audformat.Column(
            scheme_id="grbas_roughness_category"
        )
        db[table_id]["grbas_roughness_category"].set(df_split["grbas_roughness_category"])

        db[table_id]["grbas_breathiness_score"] = audformat.Column(
            scheme_id="grbas_breathiness_score"
        )
        db[table_id]["grbas_breathiness_score"].set(df_split["grbas_breathiness_score"])

        db[table_id]["grbas_breathiness_category"] = audformat.Column(
            scheme_id="grbas_breathiness_category"
        )
        db[table_id]["grbas_breathiness_category"].set(df_split["grbas_breathiness_category"])

        db[table_id]["grbas_strain_score"] = audformat.Column(
            scheme_id="grbas_strain_score"
        )
        db[table_id]["grbas_strain_score"].set(df_split["grbas_strain_score"])

        db[table_id]["grbas_strain_category"] = audformat.Column(
            scheme_id="grbas_strain_category"
        )
        db[table_id]["grbas_strain_category"].set(df_split["grbas_strain_category"])

        db[table_id]["gender"] = audformat.Column(scheme_id="gender")
        db[table_id]["gender"].set(df_split ["gender"])



    """creates splits for training and testing per speech task"""

    splits = ["train", "test"]
    speech_tasks = ["read_speech", "sustained_utterance"]
    for split in splits:
        df = db[f"segments.{split}"].get()
        
        #check unique values in speech_task column

        for speech_task in speech_tasks:
            table_id = f"segments.{split}.{speech_task}"

            df_split = df[(df["speech_task"] == speech_task)].reset_index(drop=False)

            df_split.set_index(["file", "start", "end"], inplace=True)

            db[table_id] = audformat.Table(index=df_split.index, media_id="microphone")

            db[table_id]["speaker"] = audformat.Column(scheme_id="speaker")
            db[table_id]["speaker"].set(df_split["speaker"])

            db[table_id]["age"] = audformat.Column(scheme_id="age")
            db[table_id]["age"].set(df_split["age"])

            db[table_id]["grbas_category"] = audformat.Column(
                scheme_id="grbas_category"
            )
            db[table_id]["grbas_category"].set(df_split["grbas_category"])

            db[table_id]["grbas_score"] = audformat.Column(scheme_id="grbas_score")
            db[table_id]["grbas_score"].set(df_split["grbas_score"])

            db[table_id]["grbas_aesthenia_score"] = audformat.Column(
                scheme_id="grbas_aesthenia_score"
            )
            db[table_id]["grbas_aesthenia_score"].set(df_split["grbas_aesthenia_score"])

            db[table_id]["grbas_aesthenia_category"] = audformat.Column(
                scheme_id="grbas_aesthenia_category"
            )
            db[table_id]["grbas_aesthenia_category"].set(
                df_split["grbas_aesthenia_category"]
            )

            db[table_id]["grbas_roughness_score"] = audformat.Column(
                scheme_id="grbas_roughness_score"
            )
            db[table_id]["grbas_roughness_score"].set(df_split["grbas_roughness_score"])

            db[table_id]["grbas_roughness_category"] = audformat.Column(
                scheme_id="grbas_roughness_category"
            )
            db[table_id]["grbas_roughness_category"].set(
                df_split["grbas_roughness_category"]
            )

            db[table_id]["grbas_breathiness_score"] = audformat.Column(
                scheme_id="grbas_breathiness_score"
            )
            db[table_id]["grbas_breathiness_score"].set(
                df_split["grbas_breathiness_score"]
            )

            db[table_id]["grbas_breathiness_category"] = audformat.Column(
                scheme_id="grbas_breathiness_category"
            )
            db[table_id]["grbas_breathiness_category"].set(
                df_split["grbas_breathiness_category"]
            )

            db[table_id]["grbas_strain_score"] = audformat.Column(
                scheme_id="grbas_strain_score"
            )
            db[table_id]["grbas_strain_score"].set(df_split["grbas_strain_score"])

            db[table_id]["grbas_strain_category"] = audformat.Column(
                scheme_id="grbas_strain_category"
            )
            db[table_id]["grbas_strain_category"].set(df_split["grbas_strain_category"])

            db[table_id]["gender"] = audformat.Column(scheme_id="gender")
            db[table_id]["gender"].set(df_split["gender"])


    """
        - CAPE-V score: breathiness, roughness, loudness, strain, severity, pitch, and the ratings of each rater at each time point
    """

    df_cape_pitch = pd.read_excel(os.path.join(data_path, excel_path, "cape_v_pitch_only.xlsx"))
    df_cape_pitch.rename(
        columns={
            "File": "speaker",
            "Average Formula": "cape_v_pitch_score",
            "Standard Deviation": "cape_v_pitch_std",
            "Rater 1 Time 1": "cape_v_pitch_rater_1_time_1",
            "Rater 1 time 2": "cape_v_pitch_rater_1_time_2",
            "Rater 2 Time 1": "cape_v_pitch_rater_2_time_1",
            "Rater 2 Time 2": "cape_v_pitch_rater_2_time_2",
            "Rater 3 Time 1": "cape_v_pitch_rater_3_time_1",
            "Rater 3 Time 2": "cape_v_pitch_rater_3_time_2",
        },
        inplace=True,
    )

    df_cape_breathiness = pd.read_excel(os.path.join(data_path, excel_path, "cape_v_breathiness_only.xlsx"))
    df_cape_breathiness.rename(
        columns={
            "File": "speaker",
            "Average Values": "cape_v_breathiness_score",
            "Standard Deviation": "cape_v_breathiness_std",
            "Rater 1 Time 1": "cape_v_breathiness_rater_1_time_1",
            "Rater 1 time 2": "cape_v_breathiness_rater_1_time_2",
            "Rater 2 Time 1": "cape_v_breathiness_rater_2_time_1",
            "Rater 2 Time 2": "cape_v_breathiness_rater_2_time_2",
            "Rater 3 Time 1": "cape_v_breathiness_rater_3_time_1",
            "Rater 3 Time 2": "cape_v_breathiness_rater_3_time_2",
        },
        inplace=True,
    )

    df_cape_roughness = pd.read_excel(os.path.join(data_path, excel_path, "cape_v_roughness_only.xlsx"))
    df_cape_roughness.rename(
        columns={
            "File": "speaker",
            "Average Values": "cape_v_roughness_score",
            "Standard Deviation": "cape_v_roughness_std",
            "Rater 1 Time 1": "cape_v_roughness_rater_1_time_1",
            "Rater 1 time 2": "cape_v_roughness_rater_1_time_2",
            "Rater 2 Time 1": "cape_v_roughness_rater_2_time_1",
            "Rater 2 Time 2": "cape_v_roughness_rater_2_time_2",
            "Rater 3 Time 1": "cape_v_roughness_rater_3_time_1",
            "Rater 3 Time 2": "cape_v_roughness_rater_3_time_2",
        },
        inplace=True,
    )

    df_cape_loudness = pd.read_excel(os.path.join(data_path, excel_path, "cape_v_loudness_only.xlsx"))

    df_cape_loudness.rename(
        columns={
            "File": "speaker",
            "Average Formula": "cape_v_loudness_score",
            "Standard Deviation": "cape_v_loudness_std",
            "Rater 1 Time 1": "cape_v_loudness_rater_1_time_1",
            "Rater 1 time 2": "cape_v_loudness_rater_1_time_2",
            "Rater 2 Time 1": "cape_v_loudness_rater_2_time_1",
            "Rater 2 Time 2": "cape_v_loudness_rater_2_time_2",
            "Rater 3 Time 1": "cape_v_loudness_rater_3_time_1",
            "Rater 3 Time 2": "cape_v_loudness_rater_3_time_2",
        },
        inplace=True,
    )

    df_cape_strain = pd.read_excel(os.path.join(data_path, excel_path, "cape_v_strain_only.xlsx"))
    df_cape_strain.rename(
        columns={
            "File": "speaker",
            "Average Values": "cape_v_strain_score",
            "Standard Deviation": "cape_v_strain_std",
            "Rater 1 Time 1": "cape_v_strain_rater_1_time_1",
            "Rater 1 time 2": "cape_v_strain_rater_1_time_2",
            "Rater 2 Time 1": "cape_v_strain_rater_2_time_1",
            "Rater 2 Time 2": "cape_v_strain_rater_2_time_2",
            "Rater 3 Time 1": "cape_v_strain_rater_3_time_1",
            "Rater 3 Time 2": "cape_v_strain_rater_3_time_2",
        },
        inplace=True,
    )

    df_cape_severity = pd.read_excel(os.path.join(data_path, excel_path, "cape_v_severity_only.xlsx"))
    df_cape_severity.rename(
        columns={
            "File": "speaker",
            "Average Values": "cape_v_severity_score",
            "Standard Deviation": "cape_v_severity_std",
            "Rater 1 Time 1": "cape_v_severity_rater_1_time_1",
            "Rater 1 time 2": "cape_v_severity_rater_1_time_2",
            "Rater 2 Time 1": "cape_v_severity_rater_2_time_1",
            "Rater 2 Time 2": "cape_v_severity_rater_2_time_2",
            "Rater 3 Time 1": "cape_v_severity_rater_3_time_1",
            "Rater 3 Time 2": "cape_v_severity_rater_3_time_2",
        },
        inplace=True,
    )

    # for all dataframes keep only the columns 'speaker', 'cape_v_x_score', and the ratings of each rater at each time point
    df_cape_pitch = df_cape_pitch[
        [
            "speaker",
            "cape_v_pitch_score",
            "cape_v_pitch_rater_1_time_1",
            "cape_v_pitch_rater_1_time_2",
            "cape_v_pitch_rater_2_time_1",
            "cape_v_pitch_rater_2_time_2",
            "cape_v_pitch_rater_3_time_1",
            "cape_v_pitch_rater_3_time_2",
        ]
    ]
    df_cape_breathiness = df_cape_breathiness[
        [
            "speaker",
            "cape_v_breathiness_score",
            "cape_v_breathiness_rater_1_time_1",
            "cape_v_breathiness_rater_1_time_2",
            "cape_v_breathiness_rater_2_time_1",
            "cape_v_breathiness_rater_2_time_2",
            "cape_v_breathiness_rater_3_time_1",
            "cape_v_breathiness_rater_3_time_2",
        ]
    ]
    df_cape_roughness = df_cape_roughness[
        [
            "speaker",
            "cape_v_roughness_score",
            "cape_v_roughness_rater_1_time_1",
            "cape_v_roughness_rater_1_time_2",
            "cape_v_roughness_rater_2_time_1",
            "cape_v_roughness_rater_2_time_2",
            "cape_v_roughness_rater_3_time_1",
            "cape_v_roughness_rater_3_time_2",
        ]
    ]

    df_cape_loudness = df_cape_loudness[
        [
            "speaker",
            "cape_v_loudness_score",
            "cape_v_loudness_rater_1_time_1",
            "cape_v_loudness_rater_1_time_2",
            "cape_v_loudness_rater_2_time_1",
            "cape_v_loudness_rater_2_time_2",
            "cape_v_loudness_rater_3_time_1",
            "cape_v_loudness_rater_3_time_2",
        ]
    ]
    df_cape_strain = df_cape_strain[
        [
            "speaker",
            "cape_v_strain_score",
            "cape_v_strain_rater_1_time_1",
            "cape_v_strain_rater_1_time_2",
            "cape_v_strain_rater_2_time_1",
            "cape_v_strain_rater_2_time_2",
            "cape_v_strain_rater_3_time_1",
            "cape_v_strain_rater_3_time_2",
        ]
    ]
    df_cape_severity = df_cape_severity[
        [
            "speaker",
            "cape_v_severity_score",
            "cape_v_severity_rater_1_time_1",
            "cape_v_severity_rater_1_time_2",
            "cape_v_severity_rater_2_time_1",
            "cape_v_severity_rater_2_time_2",
            "cape_v_severity_rater_3_time_1",
            "cape_v_severity_rater_3_time_2",
        ]
    ]

    # solving speakers naming convention
    df_cape_severity["speaker"] = df_cape_severity["speaker"].str.strip()
    df_cape_roughness["speaker"] = df_cape_roughness["speaker"].str.strip()
    df_cape_breathiness["speaker"] = df_cape_breathiness["speaker"].str.strip()
    df_cape_strain["speaker"] = df_cape_strain["speaker"].str.strip()
    df_cape_loudness["speaker"] = df_cape_loudness["speaker"].str.strip()
    df_cape_pitch["speaker"] = df_cape_pitch["speaker"].str.strip()

    # Merge all DataFrames on 'speaker'
    df_metadata = (
        df_cape_pitch.merge(df_cape_roughness, on="speaker", how="outer")
        .merge(df_cape_breathiness, on="speaker", how="outer")
        .merge(df_cape_strain, on="speaker", how="outer")
        .merge(df_cape_severity, on="speaker", how="outer")
        .merge(df_cape_loudness, on="speaker", how="outer")
    )

    db.schemes["cape_v_roughness_score"] = audformat.Scheme(
        audformat.define.DataType.FLOAT,
        minimum=0,
        maximum=100,
        description="Roughness score of the CAPE-V(Consensus Auditory-Perceptual Evaluation of Voice) scale. It corresponds to the perceived irregularity in the voicing source. Minimum values is 0(representing normal) and maximum value is 100 (representing the most severe impairment).",
    )
    db.schemes["cape_v_strain_score"] = audformat.Scheme(
        audformat.define.DataType.FLOAT,
        minimum=0,
        maximum=100,
        description="Strain score of the CAPE-V(Consensus Auditory-Perceptual Evaluation of Voice) scale. It corresponds to the perceived excessive vocal effort, tension, or hyperfunction. Minimum value is 0 and maximum value is 100.",
    )
    db.schemes["cape_v_loudness_score"] = audformat.Scheme(
        audformat.define.DataType.FLOAT,
        minimum=0,
        maximum=100,
        description="Loudness score of the CAPE-V(Consensus Auditory-Perceptual Evaluation of Voice) scale. It corresponds to the perceived sound intensity. Minimum values is 0(representing normal) and maximum value is 100 (representing the most severe impairment).",
    )
    db.schemes["cape_v_severity_score"] = audformat.Scheme(
        audformat.define.DataType.FLOAT,
        minimum=0,
        maximum=100,
        description="Severity score of the CAPE-V(Consensus Auditory-Perceptual Evaluation of Voice) scale. It corresponds to the perceived global voice's deviance. Minimum values is 0(representing normal) and maximum value is 100 (representing the most severe impairment).",
    )
    db.schemes["cape_v_pitch_score"] = audformat.Scheme(
        audformat.define.DataType.FLOAT,
        minimum=0,
        maximum=100,
        description="Pitch score of the CAPE-V(Consensus Auditory-Perceptual Evaluation of Voice) scale. It corresponds to the perceived fundamental frequency. Minimum values is 0(representing normal) and maximum value is 100 (representing the most severe impairment).",
    )
    db.schemes["cape_v_breathiness_score"] = audformat.Scheme(
        audformat.define.DataType.FLOAT,
        minimum=0,
        maximum=100,
        description="Breathiness score of the CAPE-V(Consensus Auditory-Perceptual Evaluation of Voice) scale. It corresponds to the audible air escape in the vocal tract. Minimum values is 0(representing normal) and maximum value is 100 (representing the most severe impairment).",
    )
    splits = ["train", "test"]
    speech_tasks = ["read_speech", "sustained_utterance"]
    for split in splits:
        for speech in speech_tasks:
            segments = db[f"segments.{split}.{speech}"].get()

            df = segments.merge(df_metadata, on="speaker")

            table_id = f"segments.{split}.{speech}"

            db[table_id]["cape_v_roughness_score"] = audformat.Column(
                scheme_id="cape_v_roughness_score"
            )
            db[table_id]["cape_v_roughness_score"].set(df["cape_v_roughness_score"])

            db[table_id]["cape_v_strain_score"] = audformat.Column(
                scheme_id="cape_v_strain_score"
            )
            db[table_id]["cape_v_strain_score"].set(df["cape_v_strain_score"])

            db[table_id]["cape_v_breathiness_score"] = audformat.Column(
                scheme_id="cape_v_breathiness_score"
            )
            db[table_id]["cape_v_breathiness_score"].set(df["cape_v_breathiness_score"])

            db[table_id]["cape_v_pitch_score"] = audformat.Column(
                scheme_id="cape_v_pitch_score"
            )
            db[table_id]["cape_v_pitch_score"].set(df["cape_v_pitch_score"])

            db[table_id]["cape_v_severity_score"] = audformat.Column(
                scheme_id="cape_v_severity_score"
            )
            db[table_id]["cape_v_severity_score"].set(df["cape_v_severity_score"])

            db[table_id]["cape_v_loudness_score"] = audformat.Column(
                scheme_id="cape_v_loudness_score"
            )
            db[table_id]["cape_v_loudness_score"].set(df["cape_v_loudness_score"])

    targets = [
        "cape_v_roughness",
        "cape_v_strain",
        "cape_v_loudness",
        "cape_v_severity",
        "cape_v_pitch",
        "cape_v_breathiness",
    ]
    for target in targets:
        files = db["files"].get()
        files.reset_index(inplace=True)

        df = files.merge(df_metadata, on="speaker")

        df.set_index("file", inplace=True)
        db.schemes[f"{target}_rater_time_1"] = audformat.Scheme(
            audformat.define.DataType.FLOAT,
            description=f"Rater Time 1 {target} score of the CAPE-V scale",
        )
        db.schemes[f"{target}_rater_time_2"] = audformat.Scheme(
            audformat.define.DataType.FLOAT,
            description=f"Rater Time 2 {target} score of the CAPE-V scale",
        )
        table_id_annotators = f"{target}.annotations"
        db[table_id_annotators] = audformat.Table(
            description=f"Annotations of the {target} score of the CAPE-V scale",
            index=df.index,
            media_id="microphone",
        )

        db[table_id_annotators][f"{target}_rater_1_time_1"] = audformat.Column(
            scheme_id=f"{target}_rater_time_1",
        )

        db[table_id_annotators][f"{target}_rater_1_time_1"].set(
            df[f"{target}_rater_1_time_1"]
        )

        db[table_id_annotators][f"{target}_rater_1_time_2"] = audformat.Column(
            scheme_id=f"{target}_rater_time_2",
        )

        db[table_id_annotators][f"{target}_rater_1_time_2"].set(
            df[f"{target}_rater_1_time_2"]
        )

        db[table_id_annotators][f"{target}_rater_2_time_1"] = audformat.Column(
            scheme_id=f"{target}_rater_time_1",
        )

        db[table_id_annotators][f"{target}_rater_2_time_1"].set(
            df[f"{target}_rater_2_time_1"]
        )

        db[table_id_annotators][f"{target}_rater_2_time_2"] = audformat.Column(
            scheme_id=f"{target}_rater_time_2",
        )

        db[table_id_annotators][f"{target}_rater_2_time_2"].set(
            df[f"{target}_rater_2_time_2"]
        )

        db[table_id_annotators][f"{target}_rater_3_time_1"] = audformat.Column(
            scheme_id=f"{target}_rater_time_1",
        )

        db[table_id_annotators][f"{target}_rater_3_time_1"].set(
            df[f"{target}_rater_3_time_1"]
        )

        db[table_id_annotators][f"{target}_rater_3_time_2"] = audformat.Column(
            scheme_id=f"{target}_rater_time_2",
        )

        db[table_id_annotators][f"{target}_rater_3_time_2"].set(
            df[f"{target}_rater_3_time_2"]
        )



    db.save(build_dir, storage_format="csv")


   
if __name__ == '__main__':
    main()
