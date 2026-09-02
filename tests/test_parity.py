"""Task 1.2/1.3 gates: model parity vs HF and lossless speculative decoding.

The tiny-config tests run everywhere and pin the *mechanism*; the marked tests
pin real-model numerics on a GPU box with downloaded weights.
"""

import os

import pytest
import torch

from treemoe.model.config import MixtralConfig
from treemoe.model.kv_cache import PagedKVCache
from treemoe.model.mixtral import MixtralForward, naive_moe
from treemoe.model.weights import LayerWeights, MixtralWeights


def random_weights(cfg: MixtralConfig, g: torch.Generator) -> MixtralWeights:
    def r(*shape, scale=0.05):
        return torch.randn(*shape, generator=g, dtype=cfg.dtype) * scale

    layers = []
    qdim = cfg.num_heads * cfg.head_dim
    kvdim = cfg.num_kv_heads * cfg.head_dim
    for _ in range(cfg.num_layers):
        layers.append(LayerWeights(
            input_layernorm=torch.ones(cfg.hidden_dim, dtype=cfg.dtype),
            post_attn_layernorm=torch.ones(cfg.hidden_dim, dtype=cfg.dtype),
            attn={
                "q_proj": r(qdim, cfg.hidden_dim),
                "k_proj": r(kvdim, cfg.hidden_dim),
                "v_proj": r(kvdim, cfg.hidden_dim),
                "o_proj": r(cfg.hidden_dim, qdim),
            },
            router=r(cfg.num_experts, cfg.hidden_dim, scale=0.2),
            w1=r(cfg.num_experts, cfg.intermediate_dim, cfg.hidden_dim),
            w2=r(cfg.num_experts, cfg.hidden_dim, cfg.intermediate_dim),
            w3=r(cfg.num_experts, cfg.intermediate_dim, cfg.hidden_dim),
        ))
    return MixtralWeights(
        config=cfg,
        embed_tokens=r(cfg.vocab_size, cfg.hidden_dim, scale=0.5),
        final_norm=torch.ones(cfg.hidden_dim, dtype=cfg.dtype),
        lm_head=r(cfg.vocab_size, cfg.hidden_dim),
        layers=layers,
    )


@pytest.fixture()
def tiny_model(tiny_config, rng):
    w = random_weights(tiny_config, rng)
    kv = PagedKVCache(tiny_config, num_blocks=8, device="cpu", dtype=tiny_config.dtype)
    return MixtralForward(w, kv, moe_fn=naive_moe)


def test_ar_incremental_matches_full_recompute(tiny_model, tiny_config, rng):
    """Paged-KV incremental decode == recomputing from scratch each step."""
    ids = torch.randint(0, tiny_config.vocab_size, (6,), generator=rng)
    pos = torch.arange(6)
    full_logits = tiny_model.forward(ids, pos)

    kv2 = PagedKVCache(tiny_config, num_blocks=8, device="cpu", dtype=tiny_config.dtype)
    m2 = MixtralForward(tiny_model.w, kv2, moe_fn=naive_moe)
    step_logits = []
    for t in range(6):
        lg = m2.forward(ids[t : t + 1], pos[t : t + 1])
        step_logits.append(lg[0])
    torch.testing.assert_close(full_logits[-1], step_logits[-1], rtol=1e-3, atol=1e-4)


def test_tree_forward_equals_path_forward(tiny_model, tiny_config, rng):
    """A linear-chain 'tree' must produce the same logits as plain AR decode."""
    prompt = torch.randint(0, tiny_config.vocab_size, (4,), generator=rng)
    tiny_model.forward(prompt, torch.arange(4))  # prefill

    chain = torch.randint(0, tiny_config.vocab_size, (3,), generator=rng)
    n = 3
    mask = torch.tril(torch.ones(n, n, dtype=torch.bool))
    positions = 4 + torch.arange(n)
    tree_logits = tiny_model.forward(chain, positions, tree_mask=mask)

    kv2 = PagedKVCache(tiny_config, num_blocks=8, device="cpu", dtype=tiny_config.dtype)
    m2 = MixtralForward(tiny_model.w, kv2, moe_fn=naive_moe)
    m2.forward(prompt, torch.arange(4))
    ar_logits = []
    for t in range(n):
        lg = m2.forward(chain[t : t + 1], positions[t : t + 1])
        ar_logits.append(lg[0])
    torch.testing.assert_close(tree_logits[-1], ar_logits[-1], rtol=1e-3, atol=1e-4)


def test_offload_staging_matches_resident(tiny_config, rng):
    """layout="offload" staging path must produce identical logits to the
    resident path (staging is a pure copy; validated here on CPU)."""
    import dataclasses

    w = random_weights(tiny_config, rng)
    ids = torch.randint(0, tiny_config.vocab_size, (5,), generator=rng)
    pos = torch.arange(5)

    kv1 = PagedKVCache(tiny_config, num_blocks=8, device="cpu", dtype=tiny_config.dtype)
    resident = MixtralForward(w, kv1, moe_fn=naive_moe).forward(ids, pos)

    w_off = dataclasses.replace(
        w, layers=[dataclasses.replace(lw, experts_on_gpu=False) for lw in w.layers])
    kv2 = PagedKVCache(tiny_config, num_blocks=8, device="cpu", dtype=tiny_config.dtype)
    staged = MixtralForward(w_off, kv2, moe_fn=naive_moe).forward(ids, pos)

    assert torch.equal(resident, staged)  # bitwise: staging is a pure copy


@pytest.mark.model
@pytest.mark.gpu
def test_ar_logits_match_hf():
    """M1 anchor (plan Task 1.2): 32 greedy steps identical to HF Mixtral.

    Runs on 24GB cards (e.g. 4090) via sequential offload: HF generates first
    with accelerate CPU-offload, is freed, then our forward re-runs with
    layout="offload" (experts pinned in host RAM, staged per layer). Slow
    (~minutes) but numerically identical -- computation stays on-GPU BF16.
    Needs host RAM >= ~110GB. Cards >= 120GB take the original resident path.
    """
    import gc

    transformers = pytest.importorskip("transformers")
    from treemoe.model.weights import default_model_dir, load_mixtral_weights

    model_dir = default_model_dir()
    tok = transformers.AutoTokenizer.from_pretrained(model_dir)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    resident = total_gb >= 120
    steps = int(os.getenv("PARITY_STEPS", "32"))
    layer_diag = os.getenv("PARITY_LAYER_DIAG") == "1"
    hf_gpu_cap = int(os.getenv("PARITY_HF_GPU_CAP_GIB", "14"))

    # ---- phase 1: HF reference tokens, then free everything ----
    print(f"[parity] phase 1/2: loading HF reference ({steps} greedy steps)", flush=True)
    hf_device_map = "cpu" if layer_diag else ("cuda" if resident else "auto")
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16,
        device_map=hf_device_map,
        max_memory=None if (resident or layer_diag) else {
            0: f"{hf_gpu_cap}GiB", "cpu": "200GiB",
        },
    )
    ids_cpu = tok("The capital of France is", return_tensors="pt").input_ids[0]
    if layer_diag:
        # transformers 5.x packs all eight experts into one ~2.8GB module, and
        # both device_map="auto" and cpu_offload accumulate enough staged
        # modules to OOM a 24GB card. Execute the unmodified HF decoder one
        # layer at a time: weights live on CPU between calls, while every
        # numerical operation still runs as BF16 on this GPU.
        from transformers.models.mixtral import modeling_mixtral

        hidden = hf.model.embed_tokens(ids_cpu.unsqueeze(0)).cuda()
        position_ids = torch.arange(ids_cpu.shape[0], device="cuda").unsqueeze(0)
        causal_mask = modeling_mixtral.create_causal_mask(
            config=hf.config, inputs_embeds=hidden, attention_mask=None,
            past_key_values=None, position_ids=position_ids,
        )
        hf.model.rotary_emb.to("cuda")
        position_embeddings = hf.model.rotary_emb(hidden, position_ids)
        hf.model.rotary_emb.to("cpu")
        hf_layers = [hidden[0].float().cpu()]
        for layer_idx, layer in enumerate(hf.model.layers):
            layer.to("cuda")
            hidden = layer(
                hidden, position_embeddings=position_embeddings,
                attention_mask=causal_mask, position_ids=position_ids,
                past_key_values=None, use_cache=False,
            )
            hf_layers.append(hidden[0].float().cpu())
            layer.to("cpu")
            torch.cuda.empty_cache()
            print(f"[parity] HF layer {layer_idx + 1}/32 ready", flush=True)
        hf_tokens, hf_scores = [], []
        print(f"[parity] HF layer references ready: {len(hf_layers)} states", flush=True)
    else:
        ids = ids_cpu.cuda()
        hf_out = hf.generate(
            ids.unsqueeze(0), do_sample=False, max_new_tokens=steps,
            return_dict_in_generate=True, output_scores=True,
        )
        hf_tokens = hf_out.sequences[0, ids.shape[0]:].tolist()
        hf_scores = [score[0].float().cpu() for score in hf_out.scores]
        print(f"[parity] HF reference ready: {len(hf_tokens)} tokens", flush=True)
    del hf
    gc.collect()
    torch.cuda.empty_cache()

    # ---- phase 2: our forward on the same prompt ----
    print("[parity] phase 2/2: loading TreeMoE offload weights", flush=True)
    cfg = MixtralConfig()
    w = load_mixtral_weights(
        model_dir, cfg,
        layout="full" if resident else "offload",
        offload_layers=None if resident else set(range(cfg.num_layers)),
    )
    kv = PagedKVCache(cfg, num_blocks=64)
    ours = MixtralForward(w, kv)

    if layer_diag:
        def compare_layer(layer_idx, state):
            # HF hidden_states[0] is embedding output; entry i+1 is decoder
            # layer i output (except the final entry may include final norm).
            ref = hf_layers[layer_idx + 1].to(state.device)
            diff = (state.float() - ref).abs()
            print(f"[parity-layer] {layer_idx:02d}: "
                  f"max_abs={float(diff.max()):.6f} "
                  f"mean_abs={float(diff.mean()):.6f}", flush=True)

        ours.layer_observer = compare_layer

    ids = ids_cpu.cuda()
    pos = torch.arange(ids.shape[0], device="cuda")
    logits = ours.forward(ids, pos)
    if layer_diag:
        pytest.fail("layer diagnostics complete (inspect [parity-layer] output)")
    our_tokens = []
    mismatches = []
    for step in range(steps):
        our_score = logits[-1].float()
        our_token = int(our_score.argmax())
        our_tokens.append(our_token)
        hf_token = hf_tokens[step]
        hf_score = hf_scores[step].to(our_score.device)
        hf_top2 = torch.topk(hf_score, 2)
        max_abs = float((our_score - hf_score).abs().max())
        hf_margin = float(hf_top2.values[0] - hf_top2.values[1])
        if our_token != hf_token:
            mismatches.append({
                "step": step,
                "ours": our_token,
                "hf": hf_token,
                "hf_margin": hf_margin,
                "max_abs": max_abs,
                "ours_at_hf": float(our_score[hf_token]),
                "ours_at_ours": float(our_score[our_token]),
                "hf_rank_of_ours": int((hf_score > hf_score[our_token]).sum()) + 1,
                "ours_rank_of_hf": int((our_score > our_score[hf_token]).sum()) + 1,
            })
        print(f"[parity] step {step + 1}/{steps}: "
              f"ours={our_token} hf={hf_token} "
              f"max_abs={max_abs:.4f} hf_margin={hf_margin:.4f}", flush=True)
        # Teacher-force the HF token so one close argmax flip cannot poison all
        # later positions; persistent errors then indicate a real forward bug.
        p = torch.tensor([kv.seq_len], device="cuda")
        token = torch.tensor([hf_token], device="cuda")
        logits = ours.forward(token, p)
    assert our_tokens == hf_tokens, f"mismatch diagnostics: {mismatches}"
