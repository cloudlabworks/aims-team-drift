"""Fetch the pinned static-embedding model into work/models/potion-base-8M and verify checksums (needed to rebuild embed_table.f16).

minishlab/potion-base-8M (MIT) at a pinned revision. The hf-xet transfer path stalled on this machine, so it is switched off
(plain HTTPS) before huggingface_hub is imported.
"""
import hashlib, os
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import hf_hub_download  # noqa: E402

REPO, REV = "minishlab/potion-base-8M", "bf8b056651a2c21b8d2565580b8569da283cab23"
SHA256 = {
    "model.safetensors": "f65d0f325faadc1e121c319e2faa41170d3fa07d8c89abd48ca5358d9a223de2",
    "tokenizer.json": "e67e803f624fb4d67dea1c730d06e1067e1b14d830e2c2202569e3ef0f70bb50",
    "vocab.txt": "1394523a67ddd404a825428018c0582a6998bcfa044ecbcbf1f4d71adb94c61c",
    "config.json": "2a6ac0e9aaa356a68a5688070db78fc3a464fefe85d2f06a1905ce3718687553",
}
OUT = "work/models/potion-base-8M"

for name, want in SHA256.items():
    path = hf_hub_download(REPO, name, revision=REV, local_dir=OUT)
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert got == want, f"{name}: sha256 {got} != pinned {want}"
    print("ok", name, got[:16])
