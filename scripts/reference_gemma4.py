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
            hidden = output[0] if isinstance(output, tuple) else output
            fingerprint(f"layer.{layer_index}", hidden)

        handles.append(layer.register_forward_hook(hook))

    with torch.inference_mode():
        output = model(**{key: value.to("cuda") for key, value in encoded.items()})
    for handle in handles:
        handle.remove()
    top = torch.topk(output.logits[0, -1].float(), 10)
    print("Gemma4 reference top10:", list(zip(top.indices.cpu().tolist(), top.values.cpu().tolist())))


if __name__ == "__main__":
    main()
