import jax
import jax.numpy as jnp
import flax.nnx as nnx

from smol_vision.models.smol_vision.config import Config
from smol_vision.models.smol_vision.smol_vision import SmolVision



@nnx.jit(static_argnums=(0, 1))
def create_sharded_model(Model, config, rngs):
    model = Model(config=config, rngs=rngs)
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = nnx.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)
    return model


def load_hf_pretrained():
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M") 
    return model


def from_hf_pretrained(
    config: Config, rngs: nnx.Rngs, sharded=False
) -> SmolVision:
    import torch
    if sharded:
        m = create_sharded_model(SmolVision, config, rngs)
    else:
        m = SmolVision(config, rngs)

    graphdef, target_params, other_state = nnx.split(m, nnx.Param, ...)

    hf_m = load_hf_pretrained()
    hf_state = hf_m.state_dict()
    

    target_params.text_decoder.wte.embedding.value = jnp.array(
        hf_state["model.embed_tokens.weight"].numpy(), dtype=config.dtype
    )
    target_params.text_decoder.rms_n_f.scale.value = jnp.array(
        hf_state["model.norm.weight"].numpy(), dtype=config.dtype
    )

    for i in range(len(target_params.text_decoder.h)):
        # MLP weights
        target_params.text_decoder.h[i].glu.gate.kernel.value = jnp.array(
            hf_state[f"model.layers.{i}.mlp.gate_proj.weight"].numpy().T,
            dtype=config.dtype,
        )
        target_params.text_decoder.h[i].glu.c_fc.kernel.value = jnp.array(
            hf_state[f"model.layers.{i}.mlp.up_proj.weight"].numpy().T,
            dtype=config.dtype,
        )
        target_params.text_decoder.h[i].glu.c_proj.kernel.value = jnp.array(
            hf_state[f"model.layers.{i}.mlp.down_proj.weight"].numpy().T,
            dtype=config.dtype,
        )

        # RMS Norm weights
        target_params.text_decoder.h[i].rms_n_1.scale.value = jnp.array(
            hf_state[f"model.layers.{i}.input_layernorm.weight"].numpy(),
            dtype=config.dtype,
        )
        target_params.text_decoder.h[i].rms_n_2.scale.value = jnp.array(
            hf_state[f"model.layers.{i}.post_attention_layernorm.weight"].numpy(),
            dtype=config.dtype,
        )

        # Causal self-attention weights
        target_params.text_decoder.h[i].attn.c_proj.kernel.value = jnp.array(
            hf_state[f"model.layers.{i}.self_attn.o_proj.weight"].numpy().T,
            dtype=config.dtype,
        )
        target_params.text_decoder.h[i].attn.wq.kernel.value = jnp.array(
            hf_state[f"model.layers.{i}.self_attn.q_proj.weight"].numpy().T,
            dtype=config.dtype,
        )
        wk = jnp.array(
            hf_state[f"model.layers.{i}.self_attn.k_proj.weight"].numpy().T,
            dtype=config.dtype,
        )
        wv = jnp.array(
            hf_state[f"model.layers.{i}.self_attn.v_proj.weight"].numpy().T,
            dtype=config.dtype,
        )
        wkv = jnp.concatenate([wk, wv], axis=1)
        target_params.text_decoder.h[i].attn.wkv.kernel.value = jnp.array(
            wkv, dtype=config.dtype
        )

    m = nnx.merge(graphdef, target_params, other_state)

    return m

