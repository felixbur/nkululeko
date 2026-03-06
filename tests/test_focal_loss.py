import configparser
import nkululeko.glob_conf as glob_conf

# Simulate config
config = configparser.ConfigParser()
config["MODEL"] = {"loss": "focal", "focal.alpha": "0.25", "focal.gamma": "2.0"}
config["DATA"] = {"target": "label"}
config["EXP"] = {"root": "test", "name": "test"}

glob_conf.config = config
glob_conf.labels = ["fake", "real"]

from nkululeko.utils.util import Util

util = Util("test")

loss_type = util.config_val("MODEL", "loss", "bce")
print(f"Loss type from config: {loss_type}")

if loss_type == "focal":
    alpha = float(util.config_val("MODEL", "focal.alpha", 0.25))
    gamma = float(util.config_val("MODEL", "focal.gamma", 2.0))
    print(f"✓ Focal loss will be used with alpha={alpha}, gamma={gamma}")
else:
    print(f"✗ Using {loss_type} loss instead")
