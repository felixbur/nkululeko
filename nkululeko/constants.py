VERSION = "1.11.5"
SAMPLING_RATE = 16000
COL_SEX = "gender"
COL_AGE = "age"
COL_SPEAKER = "speaker"
AUDIO_EXTS = ["aac", "au", "flac", "m4a", "mp3", "ogg", "wav"]

# Sentinel dict key some classifiers' predict_sample() use to carry the
# actual predicted label (from predict()) alongside per-class
# probabilities (from predict_proba()), for classifiers where those two
# can genuinely disagree (e.g. SVC(probability=True): predict_proba is a
# separate Platt-scaling fit that can re-learn its own class prior
# independently of the decision function, so argmax(predict_proba) can
# silently diverge from predict() under class_weight on imbalanced data -
# see GH #440). Consumers (nkululeko.predict) must pop this key out before
# treating the rest of the dict as per-class probability columns.
PREDICTED_LABEL_KEY = "__nkululeko_predicted_label__"
