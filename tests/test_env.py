"""Task 0.1 gate: environment + model availability (marked, skipped on dev boxes)."""

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
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto",
        # cap GPU usage so accelerate offloads to host RAM on small cards
        max_memory={0: f"{int(total_gb) - 6}GiB", "cpu": "200GiB"},
    )
    ids = tok("hello", return_tensors="pt").input_ids.to(model.device)
    out = model(ids)
    assert out.logits.shape[-1] == 32000
