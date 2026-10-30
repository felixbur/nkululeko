"""Unit tests specific to MLP_Reg_model (nkululeko/models/model_mlp_regression.py).

MLP_Reg_model and MLPModel (model_mlp.py) used to each carry their own
copy of the hidden-layer-building MLP class -- GH #467's activation-key
collision, and the layer-order/dropout-placement bugs found reviewing
its fix (PR #476), had to be found and fixed in both. They now share one
implementation, nkululeko.models.mlp_layers.MLPBlock; its architecture
tests (activation placement, dropout placement, layer order) live in
test_model_mlp.py and are not duplicated here. parse_drop_config's own
tests (including the `float(drop)` crash this file used to have, from
parsing `[MODEL] drop` inline instead of through that shared helper) live
in test_mlp_layers.py.
"""

import nkululeko.models.model_mlp as model_mlp
import nkululeko.models.model_mlp_regression as model_mlp_regression
from nkululeko.models.mlp_layers import MLPBlock


def test_classifier_and_regressor_share_one_mlp_block():
    assert model_mlp.MLPBlock is MLPBlock
    assert model_mlp_regression.MLPBlock is MLPBlock
