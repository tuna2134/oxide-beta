#!/usr/bin/env python3
"""Print layer fingerprints from the official Transformers Gemma 4 model."""

import argparse
import math

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer


def fingerprint(label, tensor):
    value = tensor.detach().float()[0, 0].cpu()
    rms = math.sqrt(torch.mean(value * value).item())
    maximum = torch.max(torch.abs(value)).item()
    print(
        f"Gemma4 reference {label}: rms={rms:.8f} abs_max={maximum:.8f} "
        f"first={value[:4].tolist()}"
    )


def hidden_output(output):
    """Extract the hidden-state tensor from module outputs such as attention."""
    return output[0] if isinstance(output, tuple) else output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("prompt")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    text = f"<start_of_turn>user\n{args.prompt}<end_of_turn>\n<start_of_turn>model\n"
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    language = model.model.language_model
    handles = []
    fingerprint("embedding", language.embed_tokens(encoded.input_ids.to("cuda")))

    for index, layer in enumerate(language.layers):
        def hook(_module, _inputs, output, layer_index=index):
            fingerprint(f"layer.{layer_index}", hidden_output(output))

        handles.append(layer.register_forward_hook(hook))

    layer_zero = language.layers[0]
    for label, module in (
        ("layer.0.input_norm", layer_zero.input_layernorm),
        ("layer.0.attention_raw", layer_zero.self_attn),
        ("layer.0.attention_norm", layer_zero.post_attention_layernorm),
        ("layer.0.pre_mlp_norm", layer_zero.pre_feedforward_layernorm),
        ("layer.0.mlp_raw", layer_zero.mlp),
        ("layer.0.mlp_norm", layer_zero.post_feedforward_layernorm),
        ("layer.0.ple_norm", layer_zero.post_per_layer_input_norm),
    ):
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, name=label: fingerprint(
                    name, hidden_output(output)
                )
            )
        )

    with torch.inference_mode():
        output = model(**{key: value.to("cuda") for key, value in encoded.items()})
    for handle in handles:
        handle.remove()
    top = torch.topk(output.logits[0, -1].float(), 10)
    print("Gemma4 reference top10:", list(zip(top.indices.cpu().tolist(), top.values.cpu().tolist())))


if __name__ == "__main__":
    main()
