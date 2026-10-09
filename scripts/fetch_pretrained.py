"""Download, or verify, the torchvision ResNet-50 IMAGENET1K_V2 weights shared by all four models."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from torch.hub import download_url_to_file
from torchvision.models import ResNet50_Weights

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-config", type=Path,
                        default=PROJECT_ROOT / "configs/formal_experiment_pid_numeric.json")
    parser.add_argument("--target", type=Path, help="Defaults to pretrained_path in the formal config")
    args = parser.parse_args()
    formal = json.loads(args.formal_config.read_text(encoding="utf-8"))
    target = args.target or PROJECT_ROOT / formal["pretrained_path"]
    expected = formal["expected_sha256"]["pretrained"]
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        if partial.exists():
            raise ValueError(f"Remove the incomplete download first: {partial}")
        url = ResNet50_Weights.IMAGENET1K_V2.url
        print(f"Downloading {url}", flush=True)
        download_url_to_file(url, str(partial), progress=True)
        if sha256(partial) != expected:
            raise ValueError(f"Downloaded file hash differs from the frozen value; kept for inspection: {partial}")
        partial.rename(target)
    actual = sha256(target)
    if actual != expected:
        raise ValueError(f"Pretrained file hash differs from the frozen value: {target}")
    print(json.dumps({"path": str(target), "sha256": actual, "verified": True}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
