"""Select and validate the two archived FastH3 decoder configurations."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parent


def provenance():
    return json.loads((ROOT / "SOURCE_PROVENANCE.json").read_text())


def fastvideo_revision():
    return provenance()["fastvideo_commit"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_variant(variant):
    source = provenance()
    if digest(ROOT / source["loader_path"]) != source["loader_sha256"]:
        raise RuntimeError("The CPU-load patch differs from the archived configuration.")
    if digest(ROOT / source["decoder_path"]) != source["decoder_sha256"][variant]:
        raise RuntimeError(
            f"Select this runner's decoder first: python publication_runtime.py {variant}"
        )


def select_variant(variant):
    source = provenance()
    actual = digest(ROOT / source["decoder_path"])
    if actual == source["decoder_sha256"][variant]:
        require_variant(variant)
        return
    if actual not in source["decoder_sha256"].values():
        raise RuntimeError("Decoder has local changes; refusing to overwrite them.")
    command = ["git", "apply", "--directory=work/FastVideo"]
    if variant == "copyback":
        command.append("--reverse")
    patch = "work/decoder-no-copyback.patch"
    subprocess.run(command + ["--check", patch], cwd=ROOT, check=True)
    subprocess.run(command + [patch], cwd=ROOT, check=True)
    require_variant(variant)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=["copyback", "no_copyback"])
    arguments = parser.parse_args()
    select_variant(arguments.variant)
    print(f"Decoder configuration: {arguments.variant}")
