# glob_conf.py


def init_config(config_obj):
    global config
    config = config_obj


def set_label_encoder(encoder):
    global label_encoder
    label_encoder = encoder


def set_util(util_obj):
    global util
    util = util_obj


def set_module(module_obj):
    global module
    module = module_obj


def set_report(report_obj):
    global report
    report = report_obj


def set_labels(labels_obj):
    global labels
    labels = labels_obj


def set_target(target_obj):
    global target
    target = target_obj
