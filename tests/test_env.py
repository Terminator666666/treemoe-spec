"""Task 0.1 gate: environment + model availability (marked, skipped on dev boxes)."""

import pytest
import torch

pytestmark = pytest.mark.model

MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"


@pytest.mark.gpu
def test_mixtral_forward_one_token():
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    ids = tok("hello", return_tensors="pt").input_ids.to(model.device)
    out = model(ids)
    assert out.logits.shape[-1] == 32000
