from typing import Annotated, Optional
from pathlib import Path

import typer

import tokenizer

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

cli: typer.Typer = typer.Typer(
    context_settings=dict(help_option_names=["-h", "--help"]),
    rich_markup_mode="rich",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


@cli.command()
def tokenize(
    text: Annotated[str, typer.Argument(help="Text to tokenize")],
    tokenizer_path: Annotated[
        Path,
        typer.Option(
            ...,
            "--tokenizer-path",
            "-t",
            path_type=Path,
            help="Path to the tokenizer file.",
        ),
    ] = Path("../rwkv_vocab_v20230424.txt"),
):
    t = tokenizer.RWKV_TOKENIZER(tokenizer_path)
    tokens = t.encode(text)
    print([t.idx2token[i] for i in tokens])
    print(tokens)


@cli.command()
def generate(
    text: Annotated[str, typer.Argument(help="Text to start from")] = "\n\n",
    tokenizer_path: Annotated[
        Path,
        typer.Option(
            ...,
            "--tokenizer-path",
            "-t",
            path_type=Path,
            help="Path to the safetensors checkpoint.",
        ),
    ] = Path("../rwkv_vocab_v20230424.txt"),
    model_path: Annotated[
        Path,
        typer.Option(
            ...,
            "--model-path",
            "-m",
            path_type=Path,
            help="Path to the tokenizer file.",
        ),
    ] = Path("../RWKV-5-World-0.1B-v1-20230803-ctx4096.safetensors"),
    max_tokens: Annotated[
        int,
        typer.Option(
            ...,
            help="Maximum amount of tokens to generate",
            min=1,
        )
    ] = 256,
):
    t = tokenizer.RWKV_TOKENIZER(tokenizer_path)
    import jax
    from safetensors.numpy import load_file
    import model

    sd = model.prepare_sd(load_file(model_path))
    n_layers = (
        max([int(i.split(".")[1]) for i in sd.keys() if i.startswith("blocks.")]) + 1
    )
    print("n_layers =", n_layers)
    embed = sd['emb.weight'].shape[-1]
    n_head = sd['blocks.0.att.time_decay'].shape[0]
    head_size = sd['blocks.0.ln1.weight'].shape[0] // n_head
    print("(embed, n_head, head_size) =", (embed, n_head, head_size))

    # print(sd["emb.weight"])

    print("Tokenizing...")
    state = model.init_state(n_layers, n_head, head_size, embed)
    tokens = t.encode(text)
    print(tokens)
    # with jax.disable_jit():
    for i in tokens[:-1]:
        state = model.forward_no_proj_jit(sd, n_layers, n_head, head_size, i, *state)[1:]
    # print(state)
    print(text, end='')
    out_t = []
    out_last = tokens[-1]
    # s = state.clone()
    # with jax.disable_jit():
    for _ in range(max_tokens):
        out, a, b, c = model.forward_jit(sd, n_layers, n_head, head_size, out_last, *state)
        state = (a, b, c)
        out_last = model.greedy_sample(out)
        # print(out_last, out)
        out_t.append(out_last)
        tmp = t.decode(out_t)
        if '\ufffd' not in tmp: # only print when we have a valid utf-8 string
            print(tmp, end="", flush=True)
            out_t = []
    print()


if __name__ == "__main__":
    cli()
