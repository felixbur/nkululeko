# Adopted from Monica Gonzales Machorro

import os
import re
import shutil

import audeer
import audformat
import pandas as pd

import preprocess


def extract_prefix(filename):
    match = re.match(r"^([A-Za-z]+[0-9]+)", filename)
    return match.group(1) if match else None


def add_col(db, table_id, col_id, data):
    db[table_id][col_id] = audformat.Column(scheme_id=col_id)
    db[table_id][col_id].set(data)


def load_grbas(path, name, avg_col, cat_col="Category Values"):
    score_col, cat_name = f"grbas_{name}_score", f"grbas_{name}_category"
    df = pd.read_excel(path).rename(columns={"File": "speaker", avg_col: score_col, cat_col: cat_name})
    df[cat_name] = df[cat_name].str.lower()
    return df[["speaker", score_col, cat_name]]


def load_cape_v(path, name, avg_col="Average Values"):
    prefix = f"cape_v_{name}"
    df = pd.read_excel(path).rename(columns={
        "File": "speaker", avg_col: f"{prefix}_score",
        "Standard Deviation": f"{prefix}_std",
        "Rater 1 Time 1": f"{prefix}_rater_1_time_1",
        "Rater 1 time 2": f"{prefix}_rater_1_time_2",
        "Rater 2 Time 1": f"{prefix}_rater_2_time_1",
        "Rater 2 Time 2": f"{prefix}_rater_2_time_2",
        "Rater 3 Time 1": f"{prefix}_rater_3_time_1",
        "Rater 3 Time 2": f"{prefix}_rater_3_time_2",
    })
    df["speaker"] = df["speaker"].str.strip()
    return df[["speaker", f"{prefix}_score"] + [c for c in df.columns if "rater" in c]]


def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = audeer.mkdir(os.path.join(root_dir, "db"))
    data_path = os.path.join(root_dir, "download_data")
    excel_path = "Ratings Spreadsheets"
    audio_path = "Audio Files"
    xls = lambda f: os.path.join(data_path, excel_path, f)

    description = (
        "The Perceptual Voice Qualities Database (PVQD) contains voice samples which have been rated "
        "by experienced voice professionals (at least 3 different raters with a minimum of 2 years' "
        "clinical experience) in order to provide educators with standardized materials to better train "
        "pre-service clinical voice professionals. It contains 296 audio files consisting of the "
        "sustained /a/ and /i/ vowels and the sentences from the Consensus Auditory-Perceptual "
        "Evaluation of Voice (CAPE-V). All recordings were made in a quiet clinical environment using "
        "a head-mounted condenser microphone at a 6-centimeter distance from the corner of the mouth "
        "and the Computerized Speech Lab (CSL) using 16-bit quantization and a sampling rate of 44.1K.."
        " Audio recordings have been edited as best as possible to remove all clinician instructions."
    )

    db = audformat.Database(
        name="pvqd-dysphonia",
        usage=audformat.define.Usage.RESEARCH,
        description=description,
        languages=[audformat.utils.map_language("en")],
        source="https://data.mendeley.com/datasets/9dz247gnyb/1",
    )

    # --- Demographics + GRBAS grade ---
    df_demo = pd.read_excel(xls("Demographics.xlsx"))
    df_demo.rename(columns={"Participant ID ": "participant_code"}, inplace=True)
    df_demo.columns = df_demo.columns.str.lower()

    file_names = audeer.list_file_names(os.path.join(data_path, audio_path), basenames=True)
    df_audios = pd.DataFrame({"file": file_names})
    df_audios["participant_code"] = df_audios["file"].map(extract_prefix)
    df_metadata = df_audios.merge(df_demo, on="participant_code")

    df_grbas = pd.read_excel(xls("grbas_grade_only.xlsx"))
    df_grbas.rename(columns={"File": "participant_code", "Average": "grbas_score", "Category": "grbas_category"}, inplace=True)
    df_grbas.columns = df_grbas.columns.str.lower()
    df_metadata = df_metadata.merge(df_grbas, on="participant_code")
    df_metadata["gender"] = df_metadata["gender"].replace({"F": "female", "M": "male"})
    df_metadata["grbas_category"] = df_metadata["grbas_category"].str.lower()
    df_metadata["file"] = df_metadata["file"].apply(lambda f: os.path.join("data/", audio_path, f))
    df_metadata.set_index("file", inplace=True)

    # --- Schemes ---
    db.media["microphone"] = audformat.Media(audformat.define.MediaType.AUDIO, format="wav")
    STR = audformat.define.DataType.STRING
    INT = audformat.define.DataType.INTEGER
    FLT = audformat.define.DataType.FLOAT
    GRBAS_LABELS = ["mild", "moderate", "severe", "normal"]
    GRBAS_DESC = "Each component is rated on an integer four point scale, in which 0 is normal, 1 slight, 2 moderate, and 3 severe"

    db.schemes["gender"] = audformat.Scheme(STR, description="Gender of the speaker")
    db.schemes["age"] = audformat.Scheme(INT, description="Age of the speaker")
    db.schemes["speaker"] = audformat.Scheme(STR, description="ID of the speaker")
    db.schemes["grbas_score"] = audformat.Scheme(FLT, description="Grade or overall severity of dysphonia. The rater examines the speaker's voice on a 4-point scale: 1, without disorder; 2, mild disorder; 3, moderated disorder; 4, severe disorder")
    db.schemes["grbas_category"] = audformat.Scheme(STR, labels=GRBAS_LABELS, description="Category for the grade or overall severity of dysphonia. This category is the results of the GRBAS score")
    for sub in ["aesthenia", "roughness", "breathiness", "strain"]:
        db.schemes[f"grbas_{sub}_score"] = audformat.Scheme(FLT, description=f"{sub.capitalize()} score of the GRBAS scale")
        db.schemes[f"grbas_{sub}_category"] = audformat.Scheme(STR, labels=GRBAS_LABELS, description=f"{sub.capitalize()} category of the GRBAS scale. {GRBAS_DESC}")
    db.schemes["speech_task"] = audformat.Scheme(STR, labels=["read_speech", "sustained_utterance"], description="type of speech task corresponding to the segment")

    CAPE_V_DESCS = {
        "roughness":   "perceived irregularity in the voicing source",
        "strain":      "perceived excessive vocal effort, tension, or hyperfunction",
        "loudness":    "perceived sound intensity",
        "severity":    "perceived global voice's deviance",
        "pitch":       "perceived fundamental frequency",
        "breathiness": "audible air escape in the vocal tract",
    }
    for name, desc in CAPE_V_DESCS.items():
        db.schemes[f"cape_v_{name}_score"] = audformat.Scheme(FLT, minimum=0, maximum=100, description=f"{name.capitalize()} score of the CAPE-V(Consensus Auditory-Perceptual Evaluation of Voice) scale. It corresponds to the {desc}. Minimum values is 0(representing normal) and maximum value is 100 (representing the most severe impairment).")
    for name in CAPE_V_DESCS:
        db.schemes[f"cape_v_{name}_rater_time_1"] = audformat.Scheme(FLT, description=f"Rater Time 1 cape_v_{name} score of the CAPE-V scale")
        db.schemes[f"cape_v_{name}_rater_time_2"] = audformat.Scheme(FLT, description=f"Rater Time 2 cape_v_{name} score of the CAPE-V scale")

    # --- Files table ---
    db["files"] = audformat.Table(index=df_metadata.index, media_id="microphone")
    for col, src in [
        ("gender",         df_metadata["gender"]),
        ("speaker",        df_metadata["participant_code"]),
        ("age",            df_metadata["age"]),
        ("grbas_category", df_metadata["grbas_category"]),
        ("grbas_score",    df_metadata["grbas_score"]),
    ]:
        add_col(db, "files", col, src)

    # --- GRBAS subscores ---
    df_grbas_sub = (
        load_grbas(xls("grbas_asthenia_only.xlsx"),   "aesthenia",  "Average Formula")
        .merge(load_grbas(xls("grbas_roughness_only.xlsx"),   "roughness",  "Average Values"),                    on="speaker", how="outer")
        .merge(load_grbas(xls("grbas_breathiness_only.xlsx"), "breathiness","Average Value", "Category Value"),   on="speaker", how="outer")
        .merge(load_grbas(xls("grbas_strain_only.xlsx"),      "strain",     "Average Values","Category Value"),   on="speaker", how="outer")
    )

    df = db["files"].get().merge(df_grbas_sub, on="speaker")
    df["gender"] = df["gender"].str.lower().map({"f": "female", "female": "female", "m": "male", "male": "male"})
    for col in [c for c in df_grbas_sub.columns if c != "speaker"] + ["gender"]:
        add_col(db, "files", col, df[col])

    # --- Copy audio files ---
    shutil.copytree(data_path, os.path.join(build_dir, "data/"), dirs_exist_ok=True)

    # --- VAD segmentation + train/test split ---
    files = db["files"].get()
    files.index = build_dir + "/" + files.index.astype(str)
    df_files = preprocess.vad_segmentation(files, strip_prefix=build_dir)
    df = preprocess.create_train_test_split(df_files)

    SEGMENT_COLS = [
        "speech_task", "speaker", "age", "grbas_category", "grbas_score",
        "grbas_aesthenia_score", "grbas_aesthenia_category",
        "grbas_roughness_score", "grbas_roughness_category",
        "grbas_breathiness_score", "grbas_breathiness_category",
        "grbas_strain_score", "grbas_strain_category", "gender",
    ]

    for split in ["train", "test"]:
        table_id = f"segments.{split}"
        df_split = df[df["split"] == split].reset_index(drop=True).set_index(["file", "start", "end"])
        db[table_id] = audformat.Table(index=df_split.index, media_id="microphone")
        for col in SEGMENT_COLS:
            add_col(db, table_id, col, df_split[col])

    for split in ["train", "test"]:
        df_seg = db[f"segments.{split}"].get()
        for speech_task in ["read_speech", "sustained_utterance"]:
            table_id = f"segments.{split}.{speech_task}"
            df_split = (df_seg[df_seg["speech_task"] == speech_task]
                        .reset_index(drop=False).set_index(["file", "start", "end"]))
            db[table_id] = audformat.Table(index=df_split.index, media_id="microphone")
            for col in SEGMENT_COLS[1:]:  # skip speech_task — it's the filter criterion
                add_col(db, table_id, col, df_split[col])

    # --- CAPE-V scores ---
    cape_files = [
        ("pitch",       "cape_v_pitch_only.xlsx",       "Average Formula"),
        ("breathiness", "cape_v_breathiness_only.xlsx", "Average Values"),
        ("roughness",   "cape_v_roughness_only.xlsx",   "Average Values"),
        ("loudness",    "cape_v_loudness_only.xlsx",    "Average Formula"),
        ("strain",      "cape_v_strain_only.xlsx",      "Average Values"),
        ("severity",    "cape_v_severity_only.xlsx",    "Average Values"),
    ]
    df_cape = None
    for name, fname, avg_col in cape_files:
        df_part = load_cape_v(xls(fname), name, avg_col)
        df_cape = df_part if df_cape is None else df_cape.merge(df_part, on="speaker", how="outer")

    for split in ["train", "test"]:
        for speech in ["read_speech", "sustained_utterance"]:
            table_id = f"segments.{split}.{speech}"
            df = db[table_id].get().merge(df_cape, on="speaker")
            for name in CAPE_V_DESCS:
                add_col(db, table_id, f"cape_v_{name}_score", df[f"cape_v_{name}_score"])

    # --- CAPE-V rater annotation tables ---
    for name in CAPE_V_DESCS:
        target = f"cape_v_{name}"
        df = db["files"].get().reset_index().merge(df_cape, on="speaker").set_index("file")
        table_id = f"{target}.annotations"
        db[table_id] = audformat.Table(
            description=f"Annotations of the {target} score of the CAPE-V scale",
            index=df.index,
            media_id="microphone",
        )
        for rater in [1, 2, 3]:
            for t in [1, 2]:
                col = f"{target}_rater_{rater}_time_{t}"
                db[table_id][col] = audformat.Column(scheme_id=f"{target}_rater_time_{t}")
                db[table_id][col].set(df[col])

    db.save(build_dir, storage_format="csv")


if __name__ == "__main__":
    main()
