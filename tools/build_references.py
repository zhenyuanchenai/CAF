"""Rebuild an ordered frozen GPT text bank from the included descriptions."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clip import clip


def main(args):
    torch.set_num_threads(1)
    metadata = json.loads(Path(args.manifest).read_text())
    names = metadata["classnames"]
    if args.split == "base":
        names = names[:metadata["base_count"]]
    descriptions = json.loads(Path(args.descriptions).read_text())
    path = args.clip_path or clip._download(clip._MODELS["ViT-B/16"])
    state = torch.jit.load(path, map_location="cpu").state_dict()
    model = clip.build_model(state, {
        "trainer": "IVLP", "vision_depth": 0, "language_depth": 0,
        "vision_ctx": 8, "language_ctx": 8
    }).float().to(args.device).eval()
    features = []
    with torch.no_grad():
        for name in names:
            tokens = clip.tokenize(descriptions[name.replace("_", " ")]).to(args.device)
            embeddings = F.normalize(model.encode_text(tokens).float(), dim=-1)
            features.append(F.normalize(embeddings.mean(dim=0), dim=-1).cpu())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"classnames": names, "features": torch.stack(features)}, output)
    print("Saved {} class prototypes to {}".format(len(names), output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--descriptions", required=True)
    parser.add_argument("--split", choices=["all", "base"], default="all")
    parser.add_argument("--output", required=True)
    parser.add_argument("--clip-path", default="")
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())

