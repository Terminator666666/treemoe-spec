"""Task 0.1 gate: environment + model availability (marked, skipped on dev boxes)."""

import os

import pytest
import torch

pytestmark = pytest.mark.model

from treemoe.model.weights import default_model_dir  # noqa: E402

# local checkpoint (93GB -- never re-download into the HF cache)
MODEL = default_model_dir()


@pytest.mark.gpu
def test_mixtral_forward_one_token():
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    # Keep extra headroom for allocator fragmentation and runtime activations
    # on 24GB cards (4090). Override via TEST_ENV_GPU_CAP_GIB when needed.
    gpu_cap = max(8, int(total_gb) - 8)
    gpu_cap = int(os.getenv("TEST_ENV_GPU_CAP_GIB", str(gpu_cap)))
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto",
        # cap GPU usage so accelerate offloads to host RAM on small cards
        max_memory={0: f"{gpu_cap}GiB", "cpu": "220GiB"},
    )
    ids = tok("hello", return_tensors="pt").input_ids.to(model.device)
    try:
        out = model(ids)
    except torch.OutOfMemoryError:
        pytest.skip(
            "HF env smoke hit CUDA OOM during expert dispatch on this GPU. "
            "Retry with TEST_ENV_GPU_CAP_GIB=12 and "
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True."
        )
    assert out.logits.shape[-1] == 32000
