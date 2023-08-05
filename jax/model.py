from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames="eps")
def ln(x: jax.Array, w: jax.Array, b: jax.Array, eps: float = 1e-5) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.var(x, axis=-1, keepdims=True)

    eps = jax.lax.convert_element_type(eps, variance.dtype)

    invariance = w * jax.lax.rsqrt(variance + eps)
    return invariance * (x - mean) + b


# TODO: verify?
# NOTE: I use flattened arrays!
@partial(jax.jit, static_argnames=("gn", "eps"))
def gn(
    x: jax.Array, w: jax.Array, b: jax.Array, gn: int, eps: float = 1e-5
) -> jax.Array:
    group_shape = (-1, gn, x.shape[0] // gn)
    # Split array into groups
    x = x.reshape(group_shape)
    # w = w.reshape(group_shape)
    # b = b.reshape(group_shape)
    mean = jnp.mean(x, -1, keepdims=True)
    variance = jnp.var(x, -1, keepdims=True)
    x = (x - mean) * jax.lax.rsqrt(variance + eps)
    x = x.flatten()

    return (x * w) + b


# Could make state donatable if we want it to be discarded
def ffn(
    x: jax.Array,
    state: jax.Array,
    ln_w: jax.Array,
    ln_b: jax.Array,
    time_mix_k: jax.Array,
    time_mix_r: jax.Array,
    kw: jax.Array,
    vw: jax.Array,
    rw: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    res = x
    x = ln(x, ln_w, ln_b)

    xk = x * time_mix_k + state * (1 - time_mix_k)
    xr = x * time_mix_r + state * (1 - time_mix_r)
    r = jax.nn.sigmoid(rw @ xr)
    k = jnp.square(jax.nn.relu(kw @ xk))
    out = (r * (vw @ k))
    # print(out)
    # __import__("sys").exit(0)

    return res + out, x


# Could make state donatable if we want it to be discarded
@partial(jax.jit, static_argnums=(0, 1))
def attn(
    H: int,
    S: int,
    x: jax.Array,
    state_x: jax.Array,
    state_h: jax.Array,
    ln_w: jax.Array,
    ln_b: jax.Array,
    time_mix_k: jax.Array,
    time_mix_v: jax.Array,
    time_mix_r: jax.Array,
    time_first: jax.Array,
    time_decay: jax.Array,
    kw: jax.Array,
    vw: jax.Array,
    rw: jax.Array,
    ow: jax.Array,
    gn_w: jax.Array,
    gn_b: jax.Array,
):
    res = x
    x = ln(x, ln_w, ln_b)

    xk = x * time_mix_k + state_x * (1 - time_mix_k)
    xv = x * time_mix_v + state_x * (1 - time_mix_v)
    xr = x * time_mix_r + state_x * (1 - time_mix_r)
    state_x = x

    r = (rw @ xr).reshape((H, 1, S))
    k = (kw @ xk).reshape((H, S, 1))
    v = (vw @ xv).reshape((H, 1, S))

    # xx = jnp.zeros((H, S), dtype=)
    a = k @ v
    xx = r @ (time_first * a + state_h)
    state_h = a + time_decay * state_h

    xx = xx.flatten()

    out = (ow @ gn(xx, gn_w, gn_b, H))
    # print(out)
    # __import__("sys").exit(0)

    return res + out, state_x, state_h


# state_x_ffn, state_x_attn, state_h
def init_state(
    n_layers: int, H: int, S: int, embed: int, dtype=jnp.float32
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return (
        jnp.zeros((n_layers, embed)),
        jnp.zeros((n_layers, embed)),
        jnp.zeros((n_layers, H, S, S)),
    )


def prepare_sd(sd: dict[jax.typing.ArrayLike]) -> dict[str, jax.typing.ArrayLike]:
    sd = {k: jax.device_put(sd[k]) for k in sd.keys()}
    if sd["blocks.0.ln0.weight"] is not None:
        sd["emb.weight"] = ln(
            sd["emb.weight"], sd.pop("blocks.0.ln0.weight"), sd.pop("blocks.0.ln0.bias")
        )
        for k in sd.keys():
            if '.time_' in k:
                sd[k] = sd[k].squeeze()
                if '.time_decay' in k:
                    sd[k] = jnp.exp(-jnp.exp(sd[k])).reshape(-1,1,1)
                if '.time_first' in k:
                    sd[k] = jnp.exp(sd[k]).reshape(-1,1,1)
    return sd

# @jax.jit
def greedy_sample(x: jax.typing.ArrayLike) -> int:
    return int(jnp.argmax(x))


def forward_no_proj(
    sd: dict[jax.typing.ArrayLike],
    n_layer: int,
    H: int,
    S: int,
    x: int,
    state_x_ffn: jax.Array,
    state_x_att: jax.Array,
    state_h: jax.Array,
):
    # ID lookups, not one-hot
    x = sd["emb.weight"][x]

    for i in range(n_layer):
        x, state_x_att_new, state_h_new = attn(
            H,
            S,
            x,
            state_x_att[i],
            state_h[i],
            sd[f"blocks.{i}.ln1.weight"],
            sd[f"blocks.{i}.ln1.bias"],
            sd[f"blocks.{i}.att.time_mix_k"],
            sd[f"blocks.{i}.att.time_mix_v"],
            sd[f"blocks.{i}.att.time_mix_r"],
            sd[f"blocks.{i}.att.time_first"],
            sd[f"blocks.{i}.att.time_decay"],
            sd[f"blocks.{i}.att.key.weight"],
            sd[f"blocks.{i}.att.value.weight"],
            sd[f"blocks.{i}.att.receptance.weight"],
            sd[f"blocks.{i}.att.output.weight"],
            sd[f"blocks.{i}.att.ln_x.weight"],
            sd[f"blocks.{i}.att.ln_x.bias"],
        )
        state_x_att = state_x_att.at[i].set(state_x_att_new)
        state_h = state_h.at[i].set(state_h_new)
        del state_x_att_new
        del state_h_new
        # ---
        x, state_x_ffn_new = ffn(
            x,
            state_x_ffn[i],
            sd[f"blocks.{i}.ln2.weight"],
            sd[f"blocks.{i}.ln2.bias"],
            sd[f"blocks.{i}.ffn.time_mix_k"],
            sd[f"blocks.{i}.ffn.time_mix_r"],
            sd[f"blocks.{i}.ffn.key.weight"],
            sd[f"blocks.{i}.ffn.value.weight"],
            sd[f"blocks.{i}.ffn.receptance.weight"],
        )
        state_x_ffn = state_x_ffn.at[i].set(state_x_ffn_new)
        del state_x_ffn_new

    return x, state_x_ffn, state_x_att, state_h

forward_no_proj_jit = jax.jit(forward_no_proj, static_argnums=(1, 2, 3))

def forward(
    sd: dict[jax.typing.ArrayLike],
    n_layer: int,
    H: int,
    S: int,
    x: int,
    state_x_ffn: jax.Array,
    state_x_att: jax.Array,
    state_h: jax.Array,
):
    x, state_x_ffn, state_x_att, state_h = forward_no_proj(
        sd, n_layer, H, S, x, state_x_ffn, state_x_att, state_h
    )

    # Output projection
    x = ln(x, sd["ln_out.weight"], sd["ln_out.bias"])
    x = sd["head.weight"] @ x
    return x, state_x_ffn, state_x_att, state_h

forward_jit = jax.jit(forward, static_argnums=(1, 2, 3))