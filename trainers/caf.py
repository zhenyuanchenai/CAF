import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import set_random_seed

from clip import clip
from datasets.caf_splits import build_manifest_loaders
from .fusion import ContextualFusion, estimate_test_statistics, fuse_test_logits


def load_clip_to_cpu(cfg, prompted=True):
    url = clip._MODELS[cfg.MODEL.BACKBONE.NAME]
    path = cfg.TRAINER.CAF.CLIP_PATH or clip._download(url)
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != url.split("/")[-2]:
        raise ValueError("The CLIP checkpoint differs from the official backbone")
    state_dict = torch.jit.load(path, map_location="cpu").eval().state_dict()
    depth = cfg.TRAINER.CAF.PROMPT_DEPTH if prompted else 0
    design = {
        "trainer": "IVLP", "vision_depth": depth, "language_depth": depth,
        "vision_ctx": cfg.TRAINER.CAF.N_CTX, "language_ctx": cfg.TRAINER.CAF.N_CTX
    }
    return clip.build_model(state_dict, design).float()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = self.transformer(x.permute(1, 0, 2))
        x = self.ln_final(x.permute(1, 0, 2)).type(self.dtype)
        indices = tokenized_prompts.argmax(dim=-1).to(x.device)
        return x[torch.arange(x.shape[0], device=x.device), indices] @ self.text_projection


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_ctx = cfg.TRAINER.CAF.N_CTX
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx = nn.Parameter(torch.empty(n_ctx, ctx_dim))
        nn.init.normal_(self.ctx, std=0.02)
        prefix = " ".join(["X"] * n_ctx)
        prompts = [prefix + " " + name.replace("_", " ") + "." for name in classnames]
        tokens = clip.tokenize(prompts)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokens).float()
        self.register_buffer("token_prefix", embedding[:, :1], persistent=False)
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:], persistent=False)
        self.register_buffer("tokenized_prompts", tokens, persistent=False)
        self.n_cls = len(classnames)

    def forward(self):
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)


def load_reference(path, classnames):
    # The ordered class vocabulary travels with each frozen text bank.
    bank = torch.load(path, map_location="cpu")
    saved = [n.replace("_", " ") for n in bank["classnames"]]
    requested = [n.replace("_", " ") for n in classnames]
    if saved == requested:
        rows = list(range(len(saved)))
    elif saved[-len(requested):] == requested:
        # Novel classes are the second half of the complete class vocabulary.
        rows = list(range(len(saved) - len(requested), len(saved)))
    else:
        raise ValueError("Reference class order does not match the test vocabulary")
    features = bank["features"][rows].float()
    if features.shape != (len(classnames), 512) or not torch.isfinite(features).all():
        raise ValueError("Invalid reference features")
    return features


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, frozen_clip):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.frozen_image_encoder = frozen_clip.visual
        self.register_buffer(
            "reference", load_reference(cfg.TRAINER.CAF.REFERENCE, classnames),
            persistent=False
        )
        self.amp = cfg.TRAINER.CAF.PREC == "amp"
        self.image_encoder.transformer.gradient_checkpointing = cfg.TRAINER.CAF.CHECKPOINTING
        self.text_encoder.transformer.gradient_checkpointing = cfg.TRAINER.CAF.CHECKPOINTING

    def text_features(self):
        with autocast(enabled=self.amp):
            text = self.text_encoder(
                self.prompt_learner(), self.prompt_learner.tokenized_prompts
            )
            text = F.normalize(text.float(), dim=-1)
        return text, F.normalize(self.reference.detach().float(), dim=-1)

    def forward(self, image, text=None):
        if text is None:
            text = self.text_features()
        t, g = text
        # Preserve the two normalization steps used for the released image features.
        with torch.no_grad(), autocast(enabled=self.amp):
            u0 = F.normalize(self.frozen_image_encoder(image), dim=-1)
            u0 = F.normalize(u0.float(), dim=-1)
        with autocast(enabled=self.amp):
            v = F.normalize(self.image_encoder(image), dim=-1)
            v = F.normalize(v.float(), dim=-1)
        with autocast(enabled=False):
            scale = self.logit_scale.float().exp().detach()
            logits = torch.stack([
                scale * (u0.float() @ g.T),
                scale * (v.float() @ g.T),
                scale * (u0.float() @ t.T),
                scale * (v.float() @ t.T),
            ], dim=1)
        return logits, t, g


def margin(logits, labels):
    mask = F.one_hot(labels, num_classes=logits.shape[-1]).bool()
    other = logits.masked_fill(mask, float("-inf")).detach().argmax(dim=-1)
    row = torch.arange(len(labels), device=labels.device)
    return logits[row, labels] - logits[row, other]


def prompt_losses(logits, text, reference, labels, lambda_margin, lambda_text):
    ce = F.cross_entropy(logits[:, 3], labels)
    frozen_margin = margin(logits[:, 0].detach(), labels)
    preserve = (
        (frozen_margin > 0).float()
        * F.relu(frozen_margin - margin(logits[:, 1], labels))
    ).mean()
    anchor = (1 - (text.float() * reference.detach().float()).sum(dim=-1)).mean()
    return ce + lambda_margin * preserve + lambda_text * anchor, ce, preserve, anchor


@TRAINER_REGISTRY.register()
class CAF(TrainerX):
    def check_cfg(self, cfg):
        if cfg.MODEL.BACKBONE.NAME != "ViT-B/16":
            raise ValueError("The released checkpoints use ViT-B/16")
        if cfg.TRAINER.CAF.PREC not in ["fp32", "amp"]:
            raise ValueError("Use fp32 or amp; published models were trained with amp")
        if cfg.TEST.READOUT not in ["fusion", "VT"]:
            raise ValueError("TEST.READOUT must be fusion or VT")
        if cfg.TEST.FINAL_MODEL != "last_step":
            raise ValueError("CAF evaluates the final epoch")
        if not cfg.TRAINER.CAF.REFERENCE:
            raise ValueError("Set TRAINER.CAF.REFERENCE to an ordered frozen text bank")

    def build_data_loader(self):
        if self.cfg.DATASET.MANIFEST:
            build_manifest_loaders(self)
        else:
            super().build_data_loader()
            self.classnames = self.dm.dataset.classnames

    def build_model(self):
        cfg = self.cfg
        print("Loading CLIP (backbone: {})".format(cfg.MODEL.BACKBONE.NAME))
        prompted = load_clip_to_cpu(cfg)
        frozen = load_clip_to_cpu(cfg, prompted=False)
        self.model = CustomCLIP(cfg, self.classnames, prompted, frozen)
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(
                name == "prompt_learner.ctx"
                or (name.startswith(("image_encoder.", "text_encoder.")) and "VPT" in name)
            )
        self.prompt_parameters = {
            n: p for n, p in self.model.named_parameters() if p.requires_grad
        }
        print("Trainable prompt parameters: {}".format(
            sum(p.numel() for p in self.prompt_parameters.values())
        ))
        if cfg.TRAINER.CAF.INIT_PROMPT:
            self.load_prompt_state(torch.load(cfg.TRAINER.CAF.INIT_PROMPT, map_location="cpu"))
        else:
            generator = torch.Generator().manual_seed(cfg.SEED)
            with torch.no_grad():
                for p in self.prompt_parameters.values():
                    p.copy_(torch.empty(p.shape).normal_(0, 0.02, generator=generator))
                photo = prompted.token_embedding(clip.tokenize("a photo of a"))[0, 1:5]
                self.model.prompt_learner.ctx[:4].copy_(photo)
        self.model.to(self.device)
        self.fusion = ContextualFusion(
            self.num_classes, cfg.TRAINER.CAF.ALPHA_INIT,
            cfg.TRAINER.CAF.BETA_INIT, cfg.TRAINER.CAF.EMA_RATE
        ).to(self.device)
        self.optim = build_optimizer(list(self.prompt_parameters.values()), cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.gate_optim = torch.optim.Adam(
            self.fusion.parameters(), lr=cfg.TRAINER.CAF.GATE_LR,
            betas=(0.9, 0.999), eps=1e-8, weight_decay=0
        )
        self.scaler = GradScaler(enabled=cfg.TRAINER.CAF.PREC == "amp")
        self.register_model("CAF", self.model, self.optim, self.sched)
        self.loaded_epoch = None

    def load_prompt_state(self, state):
        if set(state) != set(self.prompt_parameters):
            raise ValueError("Checkpoint prompt keys do not match the model")
        with torch.no_grad():
            for name, parameter in self.prompt_parameters.items():
                value = state[name]
                if value.shape != parameter.shape or not torch.isfinite(value).all():
                    raise ValueError("Invalid prompt tensor: " + name)
                parameter.copy_(value.to(parameter))

    def before_epoch(self):
        seed = self.cfg.SEED + 100003 * (
            self.epoch + self.cfg.TRAINER.CAF.EPOCH_SEED_OFFSET
        )
        set_random_seed(seed)
        peak = self.cfg.TRAINER.CAF.GATE_LR
        lr = self.cfg.TRAINER.CAF.GATE_WARMUP_LR if self.epoch == 0 else (
            peak * 0.5 * (1 + math.cos(math.pi * (self.epoch - 1) / self.max_epoch))
        )
        for group in self.gate_optim.param_groups:
            group["lr"] = lr
        self.fusion.train()
        self.epoch_metrics = []

    def parse_batch_train(self, batch):
        views = batch["img"]
        if not isinstance(views, (tuple, list)) or len(views) != 4:
            raise ValueError("CAF expects four independently augmented views")
        image = torch.cat(views, dim=0).to(self.device)
        labels = batch["label"].repeat(4).to(self.device)
        return image, labels

    def forward_backward(self, batch):
        image, labels = self.parse_batch_train(batch)
        self.optim.zero_grad(set_to_none=True)
        self.gate_optim.zero_grad(set_to_none=True)
        logits, text, reference = self.model(image)
        loss, ce, preserve, anchor = prompt_losses(
            logits, text, reference, labels,
            self.cfg.TRAINER.CAF.LAMBDA_MARGIN,
            self.cfg.TRAINER.CAF.LAMBDA_TEXT
        )
        with autocast(enabled=False):
            fused, weights, batch_mean = self.fusion(logits.detach())
            fusion_loss = F.cross_entropy(fused, labels)
        self.detect_anomaly(loss)
        self.detect_anomaly(fusion_loss)
        scale = self.scaler.get_scale()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optim)
        self.scaler.step(self.optim)
        self.scaler.update()
        fusion_loss.backward()
        self.gate_optim.step()
        self.fusion.after_step(batch_mean)
        metrics = {
            "loss": loss.item(), "ce": ce.item(), "margin": preserve.item(),
            "text": anchor.item(), "fusion_ce": fusion_loss.item(),
            "alpha": self.fusion.alpha.item(), "beta": self.fusion.beta.item(),
            "amp_skip": self.scaler.get_scale() < scale,
            "prompt_lr": self.optim.param_groups[0]["lr"],
            "gate_lr": self.gate_optim.param_groups[0]["lr"],
        }
        self.epoch_metrics.append(metrics)
        if self.batch_idx + 1 == self.num_batches:
            self.update_lr()
        return {k: v for k, v in metrics.items() if k != "amp_skip"}

    def after_epoch(self):
        path = Path(self.output_dir) / ("metrics_epoch{}.jsonl".format(self.epoch + 1))
        path.write_text("".join(json.dumps(r) + "\n" for r in self.epoch_metrics))
        self.save_model(self.epoch, self.output_dir)

    def save_model(self, epoch, directory, **kwargs):
        folder = Path(directory) / "CAF"
        folder.mkdir(parents=True, exist_ok=True)
        scheduler = self.sched.state_dict()
        successor = scheduler.pop("successor", None)
        if successor is not None:
            scheduler["successor_state"] = successor.state_dict()
        checkpoint = {
            "state_dict": {n: p.detach().cpu() for n, p in self.prompt_parameters.items()},
            "fusion": {k: v.detach().cpu() for k, v in self.fusion.state_dict().items()},
            "epoch": epoch + 1,
            "optimizer": self.optim.state_dict(), "scheduler": scheduler,
            "gate_optimizer": self.gate_optim.state_dict(), "scaler": self.scaler.state_dict(),
        }
        path = folder / ("model.pth.tar-{}".format(epoch + 1))
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, tmp)
        tmp.replace(path)

    def resume_model_if_exist(self, directory):
        files = [p for p in (Path(directory) / "CAF").glob("model.pth.tar-*")
                 if p.name.rsplit("-", 1)[-1].isdigit()]
        if not files:
            return 0
        path = max(files, key=lambda p: int(p.name.rsplit("-", 1)[-1]))
        ck = torch.load(path, map_location="cpu")
        if "optimizer" not in ck:
            raise ValueError("Released inference weights cannot resume an optimizer; use --eval-only")
        self.load_prompt_state(ck["state_dict"])
        self.fusion.load_state_dict(ck["fusion"])
        self.optim.load_state_dict(ck["optimizer"])
        scheduler = dict(ck["scheduler"])
        successor = scheduler.pop("successor_state", None)
        self.sched.load_state_dict(scheduler)
        if successor is not None:
            self.sched.successor.load_state_dict(successor)
        self.gate_optim.load_state_dict(ck["gate_optimizer"])
        self.scaler.load_state_dict(ck["scaler"])
        for optim in [self.optim, self.gate_optim]:
            for state in optim.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        print("Resuming after epoch {}".format(ck["epoch"]))
        return ck["epoch"]

    def load_model(self, directory, epoch=None):
        if not directory or epoch is None:
            raise ValueError("Evaluation needs --model-dir and --load-epoch")
        path = Path(directory) / "CAF" / ("model.pth.tar-{}".format(epoch))
        ck = torch.load(path, map_location="cpu")
        if ck["epoch"] != epoch:
            raise ValueError("Requested epoch does not match checkpoint")
        self.load_prompt_state(ck["state_dict"])
        with torch.no_grad():
            for name in ["alpha", "beta"]:
                value = ck["fusion"][name]
                if not torch.isfinite(value).all() or value.item() < 0:
                    raise ValueError("Invalid learned fusion scalar")
                getattr(self.fusion, name).copy_(value.to(self.device))
        # Training EMA is intentionally not loaded for target-set inference.
        self.loaded_epoch = epoch
        print("Loaded {} (epoch {})".format(path, epoch))

    @torch.no_grad()
    def test(self, split=None):
        self.model.eval()
        self.fusion.eval()
        folder = Path(self.output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        n = len(self.test_loader.dataset)
        classes = len(self.classnames)
        logits = np.lib.format.open_memmap(
            folder / "logits.npy", mode="w+", dtype=np.float32, shape=(n, 4, classes)
        )
        labels, paths = [], []
        text = self.model.text_features()
        position = 0
        for batch in self.test_loader:
            image = batch["img"].to(self.device)
            z, _, _ = self.model(image, text=text)
            count = len(image)
            logits[position:position + count] = z.cpu().numpy()
            position += count
            labels.extend(batch["label"].tolist())
            paths.extend(batch["impath"])
        if position != n:
            raise ValueError("Incomplete test collection")
        logits.flush()
        h, q, mean_probability = estimate_test_statistics(logits)
        predictions = []
        weight_sum = np.zeros(4, dtype=np.float64)
        self.evaluator.reset()
        for start in range(0, n, 256):
            output, weights = fuse_test_logits(
                logits[start:start + 256], h[start:start + 256], q[start:start + 256],
                self.fusion.alpha, self.fusion.beta, device=self.device
            )
            if self.cfg.TEST.READOUT == "VT":
                output = torch.as_tensor(
                    np.array(logits[start:start + 256, 3]), device=self.device
                )
            y = torch.as_tensor(labels[start:start + len(output)], device=self.device)
            self.evaluator.process(output, y)
            predictions.extend(output.argmax(dim=-1).cpu().tolist())
            weight_sum += weights.double().sum(dim=0).cpu().numpy()
        self.evaluator.evaluate()
        correct = int(np.sum(np.asarray(predictions) == np.asarray(labels)))
        result = dict(
            dataset=self.cfg.DATASET.NAME, split=self.cfg.DATASET.SUBSAMPLE_CLASSES,
            readout=self.cfg.TEST.READOUT, seed=self.cfg.SEED,
            epoch=self.loaded_epoch or self.max_epoch, n=n, correct=correct,
            accuracy=100.0 * correct / n, alpha=self.fusion.alpha.item(),
            beta=self.fusion.beta.item(), mean_weights=(weight_sum / n).tolist(),
            route_order=["00", "V0", "0T", "VT"]
        )
        np.save(folder / "predictions.npy", np.asarray(predictions, dtype=np.int64))
        np.save(folder / "shared_mean_probability.npy", mean_probability)
        (folder / "classes.json").write_text(json.dumps(self.classnames))
        (folder / "paths.json").write_text(json.dumps(paths))
        (folder / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        return result["accuracy"]

