# global_config.py

def init_config(config_obj):
    global config
    config = config_obj

def set_label_encoder(encoder):
    global label_encoder
    label_encoder = encoder