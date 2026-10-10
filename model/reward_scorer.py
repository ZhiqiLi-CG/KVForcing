"""Frozen VideoReward scorer for Re-DMD (Reward Forcing, arXiv 2512.04678; code github.com/JaydenLyh/Reward-Forcing @4317f5a).

Re-DMD multiplies the generator's DMD loss of each sample by exp(beta * r), r = the standardized motion-quality (MQ) score
of the generated video by the frozen VideoReward model (KlingTeam/VideoReward, a Qwen2-VL-2B reward model). The score is a
constant weight: it is computed under no_grad from a no_grad VAE decode, so no gradient flows through the reward model.

The model is loaded with the unchanged videoalign code in third_party/ (trl is replaced by third_party/trl_stub, which only
covers what inference needs), with its base model read from a local directory instead of the Hugging Face hub.

preprocess (how the decoded clip [B, T, 3, H, W] in [-1, 1] is turned into the frames VideoReward sees):
  "rf"     exactly as Reward Forcing trains (model/re_dmd.py:148): frames = 255 * pixels -- i.e. still in [-255, 255]
           -- and all T frames (its frame subsampling is commented out in videoalign/vision_process.py:300-303);
  "fixed"  as VideoReward was trained (videoalign/data.py:206: frames in [0, 1]; model_config fps 2.0):
           frames = 255 * (pixels * 0.5 + 0.5) in [0, 255], uniformly subsampled to round(T / 16 fps * 2 fps) frames
           (an even number, Qwen2-VL's temporal patch is 2).
Both then resize to 336 x 588 (process_wanvideo_tensor) and let the processor rescale by 1/255, as Reward Forcing does.

Deliberately NOT an nn.Module: it is never FSDP-wrapped, never in an optimizer, never in a checkpoint or the EMA."""
import os
import sys

import torch

_TP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party")


class RewardScorer:
    def __init__(self, reward_path, base_path, device, preprocess="fixed", key="MQ", dtype=torch.bfloat16, fps=16.0):
        assert preprocess in ("rf", "fixed"), preprocess
        assert key in ("VQ", "MQ", "TA", "Overall"), key
        for p in (os.path.join(_TP, "trl_stub"), _TP):
            if p not in sys.path:
                sys.path.insert(0, p)
        from videoalign.data import DataConfig
        from videoalign.inference import VideoVLMRewardInference, load_configs_from_json
        from videoalign.train_reward import create_model_and_processor
        from videoalign.utils import ModelConfig, PEFTLoraConfig, TrainingConfig

        # same steps as VideoVLMRewardInference.__init__ (third_party/videoalign/inference.py), except that the base
        # model (model_config.json: "Qwen/Qwen2-VL-2B-Instruct") is read from base_path instead of the hub
        data_config, _, model_config, peft_lora_config, inference_config = load_configs_from_json(
            os.path.join(reward_path, "model_config.json"))
        assert os.path.isdir(base_path), base_path
        model_config["model_name_or_path"] = base_path
        training_args = TrainingConfig(load_from_pretrained=reward_path, load_from_pretrained_step=-1,
                                       gradient_checkpointing=False, disable_flash_attn2=False,
                                       bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16, output_dir="")
        model, processor, _ = create_model_and_processor(model_config=ModelConfig(**model_config),
                                                         peft_lora_config=PEFTLoraConfig(**peft_lora_config),
                                                         training_args=training_args)
        # The weights: videoalign's load_model_from_checkpoint renames the keys of model.pth from the pre-4.52 transformers
        # layout of Qwen2-VL (base_model.model.model.* = language model, base_model.model.visual.*) to the >= 4.52 layout
        # (...model.language_model.*, ...model.visual.*), because Reward Forcing runs a newer transformers. Our training env
        # has transformers 4.51.3, i.e. the pre-4.52 layout model.pth was saved in, so its keys are loaded unchanged, strictly.
        ckpts = sorted((d for d in os.listdir(reward_path) if d.startswith("checkpoint-")), key=lambda d: int(d.split("-")[-1]))
        step = ckpts[-1].split("-")[-1]
        sd = torch.load(os.path.join(reward_path, ckpts[-1], "model.pth"), map_location="cpu", weights_only=True)
        res = model.load_state_dict(sd, strict=True)
        print("[reward] loaded %s/%s/model.pth: %d tensors, missing %d, unexpected %d"
              % (reward_path, ckpts[-1], len(sd), len(res.missing_keys), len(res.unexpected_keys)), flush=True)
        del sd
        model.requires_grad_(False)
        model.eval()
        inf = VideoVLMRewardInference.__new__(VideoVLMRewardInference)
        inf.model, inf.processor, inf.device = model.to(device), processor, device
        inf.data_config, inf.inference_config = DataConfig(**data_config), inference_config
        self.inf, self.preprocess, self.key, self.fps = inf, preprocess, key, fps
        self.data_fps = float(inf.data_config.fps)          # 2.0: the frame rate VideoReward was trained on
        self.checkpoint_step = step
        print("[reward] VideoReward checkpoint-%s loaded (base %s), key %s, preprocess %s, norm %s"
              % (step, base_path, key, preprocess, {k: v for k, v in inference_config.items() if k.startswith(key)}),
              flush=True)

    def frames(self, pixels):
        """[B, T, 3, H, W] in [-1, 1] -> list of B uint8 tensors [T', 3, 336, 588] -- what VideoReward saw in training:
        frames of a decoded video (uint8), uniformly sampled at 2 fps, resized as process_wanvideo_tensor does."""
        from videoalign.vision_process import process_wanvideo_tensor
        if self.preprocess == "rf":
            # exactly Reward Forcing (model/re_dmd.py:148, videoalign/inference.py:146-149): 255 * pixels -- values in
            # [-255, 255], unclamped -- all T frames, process_wanvideo_tensor's bicubic resize to 336 x 588, float.
            # score() then lets the processor rescale (1/255) and normalize it, but skips the processor's own resize
            # (do_resize False): 336 x 588 is already its target size (multiples of 28), and in transformers 4.51.3 that
            # resize goes through PIL, which rejects values outside [0, 255].
            return [process_wanvideo_tensor(255.0 * v).float().cpu() for v in pixels.float()]
        out = []
        for v in pixels.float():
            t = v.shape[0]
            n = max(2, int(round(t / self.fps * self.data_fps)))
            n += n % 2
            idx = torch.linspace(0, t - 1, n, device=v.device).round().long()
            f = process_wanvideo_tensor(255.0 * (v[idx] * 0.5 + 0.5).clamp(0, 1))    # bicubic + antialias, as RF
            out.append(f.clamp(0, 255).round().to(torch.uint8).cpu())                # overshoot clipped, like a video
        return out

    @torch.no_grad()
    def score(self, pixels, prompts):
        """Standardized rewards (VideoReward's own mean / std) of each clip: dict key -> float32 tensor [B].
        Same as videoalign's reward_from_frames (prompt template, processor call with do_rescale, logits [B, 3], norm),
        but with the frames prepared by self.frames (uint8) instead of process_wanvideo_tensor's float output."""
        from videoalign.prompt_template import build_prompt
        assert len(prompts) == pixels.shape[0], (len(prompts), pixels.shape)
        dc = self.inf.data_config
        chat = [[{"role": "user", "content": [{"type": "video", "video": "file://dummy_path"},
                                              {"type": "text", "text": build_prompt(p, dc.eval_dim, dc.prompt_template_type)}]}]
                for p in prompts]
        batch = self.inf.processor(text=self.inf.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True),
                                   images=None, videos=self.frames(pixels), padding=True, return_tensors="pt",
                                   videos_kwargs={"do_rescale": True, **({"do_resize": False} if self.preprocess == "rf" else {})})
        logits = self.inf.model(**self.inf._prepare_inputs(batch), return_dict=True)["logits"]      # [B, 3] = VQ, MQ, TA
        ic = self.inf.inference_config
        out = {k: (logits[:, i].float() - ic[k + "_mean"]) / ic[k + "_std"] for i, k in enumerate(("VQ", "MQ", "TA"))}
        out["Overall"] = out["VQ"] + out["MQ"] + out["TA"]
        return {k: v.reshape(-1) for k, v in out.items()}

    def __call__(self, pixels, prompts):
        return self.score(pixels, prompts)[self.key]
