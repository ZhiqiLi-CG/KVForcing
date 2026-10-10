# Minimal stand-in for the `trl` package, ONLY so that third_party/videoalign can be imported for inference with the
# frozen VideoReward model. videoalign imports trl for reward-model *training* (RewardTrainer) and for k-bit quantization
# helpers; neither is used at inference (VideoReward's model_config has load_in_8bit / load_in_4bit = false).
class RewardTrainer:  # base class of videoalign's training-only VideoVLMRewardTrainer; never instantiated here
    def __init__(self, *a, **k):
        raise RuntimeError("trl stub: reward-model training is not supported in this environment")


def get_quantization_config(model_config):  # quantization is off for VideoReward
    if getattr(model_config, "load_in_8bit", False) or getattr(model_config, "load_in_4bit", False):
        raise RuntimeError("trl stub: k-bit quantization requested but not available")
    return None


def get_kbit_device_map():
    return None
