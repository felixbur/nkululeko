"""MLP_Reg_model now shares MLPBlock with MLPModel; its architecture
tests live in test_model_mlp.py, not duplicated here."""

import nkululeko.models.model_mlp as model_mlp
import nkululeko.models.model_mlp_regression as model_mlp_regression
from nkululeko.models.mlp_layers import MLPBlock


def test_classifier_and_regressor_share_one_mlp_block():
    assert model_mlp.MLPBlock is MLPBlock
    assert model_mlp_regression.MLPBlock is MLPBlock
