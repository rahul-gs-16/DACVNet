from models.acv import ACVNet

from models.acv_rest_b2_h4 import ACV_Chan_Rest_Net_b2h4

from models.loss import model_loss_train_attn_only, model_loss_train_freeze_attn, model_loss_train, model_loss_test
from models.loss_ss import model_loss_train_ss,model_loss_test_ss
from models.loss_ss import model_loss_train_ss_apply_disparity,model_loss_test_ss_apply_disparity

import models.layers_for_ss
from models.layers_for_ss import Synthesize_view_hamlyn,Synthesize_view_hamlyn2
from models.layers_for_ss import Synthesize_view_scared,apply_disparity

import models.restormer

__models__ = {
    "acvnet": ACVNet,
    "acv_rest_b2h4":ACV_Chan_Rest_Net_b2h4,

}
