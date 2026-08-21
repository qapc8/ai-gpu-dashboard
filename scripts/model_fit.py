"""What it takes to self-host the models people are actually using.

The previous version sized hardware for Llama 3.1, Qwen 2.5, Mixtral and
Gemma 2 from a hardcoded architecture table. Those are 2024 models and not one
of them appears anywhere in current demand, so the section answered a question
nobody was asking. The roster now comes from the open-weights models on the
inference leaderboard, and their architecture comes from the source:

    parameters   HuggingFace safetensors index (`safetensors.total`) -- the
                 exact count from the shipped weights, not a rounded label.
    architecture config.json: layers, KV heads, head dim, MoE expert counts.

From those, VRAM is arithmetic:

    weights   = params x bytes_per_param
    kv_cache  = 2 x layers x kv_heads x head_dim x context x 2 bytes
    overhead  = 20% of weights (activations, fragmentation, CUDA context)

The 20% is the one assumption and it is labelled as such. Mixture-of-experts
models hold every expert in VRAM even though only a few fire per token, so
weights use the total parameter count while throughput is governed by the
active count; both are reported.

The last column is the part nobody else publishes: this dashboard prices both
sides of the make-or-buy trade, so for each model we can state the output
throughput a rig must sustain to beat the cheapest API route serving that same
model. No invented tokens/sec -- the throughput is the unknown being solved for.
"""

import json
import urllib.request

GIB = 1024 ** 3
HF_API = "https://huggingface.co/api/models/{id}"
HF_RAW = "https://huggingface.co/{id}/raw/main/config.json"

# bytes per parameter
PRECISIONS = {"fp16": 2.0, "fp8": 1.0, "int4": 0.5}
CONTEXTS = [8192, 32768, 131072]

# Beyond this many GPUs a single-node fit stops being the right question.
MAX_GPUS = 8

_UA = {"User-Agent": "Mozilla/5.0 (gpu-dashboard model-fit)"}


def _get(url, timeout=30):
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _first(cfg, *names, default=None):
    for n in names:
        if cfg.get(n) is not None:
            return cfg[n]
    return default


def fetch_architecture(hf_id):
    """Exact parameter count and attention shape for a HuggingFace model."""
    info = _get(HF_API.format(id=hf_id))
    total_params = ((info.get("safetensors") or {}).get("total")) or None

    cfg = _get(HF_RAW.format(id=hf_id))
    # Multimodal repos nest the language model config; the text tower is what
    # holds the KV cache we are sizing.
    if "text_config" in cfg and isinstance(cfg["text_config"], dict):
        cfg = {**cfg, **cfg["text_config"]}

    layers = _first(cfg, "num_hidden_layers", "n_layer", "num_layers")

    # Hybrid Mamba-Transformer models (Nemotron 3 Ultra and friends) declare no
    # num_hidden_layers; they list a per-layer block type instead. Only the
    # attention blocks hold a KV cache -- Mamba blocks carry a fixed-size
    # recurrent state that does not grow with context. Nemotron 3 Ultra is 108
    # layers of which 12 are attention, so charging it for all 108 would
    # overstate its KV cache ninefold.
    block_types = cfg.get("layers_block_type")
    attention_layers = None
    if isinstance(block_types, list) and block_types:
        layers = layers or len(block_types)
        attention_layers = sum(1 for b in block_types if "attention" in str(b).lower())

    kv_heads = _first(cfg, "num_key_value_heads", "num_attention_heads", "n_head")
    heads = _first(cfg, "num_attention_heads", "n_head")
    hidden = _first(cfg, "hidden_size", "n_embd")
    head_dim = _first(cfg, "head_dim")
    if head_dim is None and hidden and heads:
        head_dim = hidden // heads

    if not (total_params and layers and kv_heads and head_dim):
        raise RuntimeError(f"{hf_id}: incomplete architecture")

    experts = _first(cfg, "n_routed_experts", "num_local_experts", "num_experts")
    active_experts = _first(cfg, "num_experts_per_tok")

    return {
        "hf_id": hf_id,
        "params": int(total_params),
        "params_b": round(total_params / 1e9, 1),
        "layers": int(layers),
        # Layers that actually hold a KV cache. Equal to `layers` for a plain
        # transformer; a fraction of it for a Mamba hybrid.
        "attention_layers": int(attention_layers) if attention_layers else int(layers),
        "kv_heads": int(kv_heads),
        "head_dim": int(head_dim),
        "attention_heads": int(heads) if heads else None,
        "experts": int(experts) if experts else None,
        "active_experts": int(active_experts) if active_experts else None,
        "max_context_k": round(_first(cfg, "max_position_embeddings", default=0) / 1000) or None,
        "model_type": cfg.get("model_type"),
    }


def _cheapest_gpu_rate(providers, gpu_id):
    best = None
    for pk, prov in (providers or {}).items():
        info = (prov.get("gpus") or {}).get(gpu_id) or {}
        price = info.get("price_per_gpu_hr")
        if price and price > 0 and (best is None or price < best[0]):
            best = (price, prov.get("provider_name", pk))
    return best


# Architectures with native FP8 tensor cores. Hopper and Blackwell added it on
# the NVIDIA side; CDNA 3 on AMD's. Ampere, Turing and CDNA 2 have no FP8 path,
# so quoting "4x T4 at fp8" as the cheapest way to serve a model is not a cheap
# option, it is not an option -- the hardware cannot execute the format.
_FP8_ARCHS = ("hopper", "blackwell", "cdna 3")
# INT4 weight-only quantisation runs anywhere with INT8/INT4 tensor cores,
# which covers Turing onward.
_NO_INT4_ARCHS = ()


def _supports(spec, precision):
    arch = (spec.get("arch") or "").lower()
    if precision == "fp8":
        return any(a in arch for a in _FP8_ARCHS)
    if precision == "int4":
        return not any(a in arch for a in _NO_INT4_ARCHS)
    return True  # fp16 is universal on everything tracked here


def _fit_options(total_gib, specs, providers, precision="fp16"):
    """Which tracked GPUs hold `total_gib` and can run `precision`, cheapest first."""
    out = []
    for gpu_id, spec in specs.items():
        vram = spec.get("vram_gb")
        if not vram:
            continue
        if not _supports(spec, precision):
            continue
        n = 1 if total_gib <= vram else int(-(-total_gib // vram))
        # Round up to a power of two. Tensor parallelism splits attention heads
        # across GPUs, so real deployments come in 1/2/4/8 -- "5x A100" is not a
        # configuration anyone can actually run, and quoting its price as the
        # cheapest option understates what the model costs to serve.
        n = 1 << (n - 1).bit_length() if n > 1 else 1
        if n > MAX_GPUS:
            continue
        rate = _cheapest_gpu_rate(providers, gpu_id)
        if not rate:
            continue
        price, prov = rate
        out.append({
            "gpu": gpu_id,
            "gpus_needed": n,
            "vram_total_gb": round(vram * n, 1),
            "headroom_pct": round((vram * n - total_gib) / (vram * n) * 100, 1),
            "cheapest_provider": prov,
            "usd_per_hr": round(price * n, 3),
        })
    out.sort(key=lambda f: (f["usd_per_hr"], f["gpus_needed"]))
    return out


def build_model_fit(data, log_info=lambda m: None, log_ok=lambda a, b="": None):
    """Size the open-weights models currently in demand against live GPU prices."""
    specs = data.get("specs") or {}
    providers = data.get("providers") or {}
    inference = data.get("inference") or {}
    models = inference.get("models") or []
    if not models or not specs:
        return data

    # One entry per model, most-used first, open weights only: you cannot
    # self-host what nobody published.
    wanted, seen = [], set()
    for m in models:
        if not m.get("open_weights") or not m.get("hf_id"):
            continue
        if m["hf_id"] in seen:
            continue
        seen.add(m["hf_id"])
        wanted.append(m)

    out, failed = {}, []
    for m in wanted:
        try:
            arch = fetch_architecture(m["hf_id"])
        except Exception:
            failed.append(m["hf_id"])
            continue

        entry = {
            "name": m["name"],
            "permaslug": m["permaslug"],
            "hf_id": arch["hf_id"],
            "params_b": arch["params_b"],
            "layers": arch["layers"],
            "attention_layers": arch["attention_layers"],
            "kv_heads": arch["kv_heads"],
            "head_dim": arch["head_dim"],
            "experts": arch["experts"],
            "active_experts": arch["active_experts"],
            "max_context_k": arch["max_context_k"],
            "model_type": arch["model_type"],
            "tokens_per_day": m.get("total_tokens"),
            "demand_rank": m.get("rank"),
            # The cheapest published route for this exact model, from the
            # inference dispersion data. This is what self-hosting has to beat.
            "api_out_per_m": m.get("out_low"),
            "api_in_per_m": m.get("in_low"),
            "api_provider": m.get("cheapest_provider"),
            "precisions": {},
        }
        if arch["experts"] and arch["active_experts"] and arch["params_b"]:
            # Rough active share: routed experts dominate parameter count.
            entry["active_share_pct"] = round(arch["active_experts"] / arch["experts"] * 100, 1)

        for prec, nbytes in PRECISIONS.items():
            weights_gib = arch["params"] * nbytes / GIB
            overhead_gib = weights_gib * 0.20
            per_ctx = {}
            for ctx in CONTEXTS:
                # KV cache stays at fp16 even when weights are quantised.
                kv_gib = (2 * arch["attention_layers"] * arch["kv_heads"]
                          * arch["head_dim"] * ctx * 2) / GIB
                total = weights_gib + overhead_gib + kv_gib
                fits = _fit_options(total, specs, providers, prec)
                rec = {
                    "kv_cache_gib": round(kv_gib, 2),
                    "total_vram_gib": round(total, 2),
                    "options": fits[:6],
                    "cheapest_usd_per_hr": fits[0]["usd_per_hr"] if fits else None,
                }
                # Output tokens/sec the cheapest rig must sustain to match the
                # cheapest API route. Solving for throughput avoids inventing it.
                if fits and entry["api_out_per_m"]:
                    rec["breakeven_tok_s"] = round(
                        fits[0]["usd_per_hr"] / entry["api_out_per_m"] * 1e6 / 3600
                    )
                per_ctx[str(ctx)] = rec
            entry["precisions"][prec] = {
                "weights_gib": round(weights_gib, 2),
                "overhead_gib": round(overhead_gib, 2),
                "contexts": per_ctx,
            }
        out[m["name"]] = entry

    if failed:
        log_info(f"Model Fit: no usable architecture for {', '.join(failed[:5])}")
    if not out:
        return data

    data["modelfit"] = out
    # The editorial workload shortlists this section used to lead with are gone:
    # hand-picked GPU lists, invented min_gpus and budget bands, and a
    # "best_value" string that named a provider which had not been cheapest for
    # months. Nothing sourced them and nothing kept them true.
    data.pop("workloads", None)
    data.pop("workload_recs", None)

    meta = data.setdefault("_meta", {}).setdefault("sections", {})
    meta["modelfit"] = {
        "basis": "derived",
        "detail": (
            "Open-weights models from the current demand leaderboard, sized "
            "against live GPU prices. Parameter counts are the exact totals "
            "from each model's HuggingFace safetensors index; layers, KV heads "
            "and head dim come from its config.json. VRAM = weights + KV cache "
            "+ 20% overhead for activations and fragmentation -- that 20% is "
            "the only assumption. MoE models hold every expert in VRAM, so "
            "weights use total parameters while the active count is reported "
            "separately. Break-even throughput solves for the output tokens/sec "
            "a rig must sustain to beat the cheapest API route for the same "
            "model, so no throughput figure is invented."
        ),
    }
    for key in ("workloads", "workload_recs"):
        meta.pop(key, None)

    log_ok("Model Fit", f"{len(out)} in-demand models sized from published architectures")
    return data
