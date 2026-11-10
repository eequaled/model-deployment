"""Which prediction does a causal decoder make from a prompt of length N?

Run this with torch in .venv-onnx. It decides whether the new prefill-on-one-position
graph can match the old whole-prompt prefill, and it is cheaper than re-exporting
blind. Answer goes to results/decide.log.
"""

import json
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.modeling_outputs import BaseModelOutput

REPO = "microsoft/Florence-2-base"
REV = "5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac"
IMG = "funsd/dataset/testing_data/images/82837252.png"


def main() -> int:
    processor = AutoProcessor.from_pretrained(REPO, revision=REV, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        REPO, revision=REV, trust_remote_code=True,
        torch_dtype=torch.float32, use_safetensors=True,
    ).to("cpu").eval()
    lm = model.language_model

    pins = processor(text="<OCR>", images=Image.open(IMG).convert("RGB"), return_tensors="pt")
    out = {}
    with torch.inference_mode():
        feats = model._encode_image(pins["pixel_values"])
        prompt_embeds = lm.get_input_embeddings()(pins["input_ids"])
        merged = torch.cat([feats, prompt_embeds], dim=1)
        mask = torch.ones(merged.shape[:2], dtype=torch.float32)
        enc = lm.model.encoder(inputs_embeds=merged, attention_mask=mask).last_hidden_state

        # (a) the old build: prefill on ALL prompt positions, keep the last logits row
        full = lm(encoder_outputs=BaseModelOutput(last_hidden_state=enc), attention_mask=mask,
                  decoder_inputs_embeds=prompt_embeds, use_cache=True, return_dict=True)
        full_tok = int(full.logits[0, -1, :].argmax())

        # (b) the new build: prefill on the LAST prompt position only
        last = lm(encoder_outputs=BaseModelOutput(last_hidden_state=enc), attention_mask=mask,
                  decoder_inputs_embeds=prompt_embeds[:, -1:, :], use_cache=True, return_dict=True)
        last_tok = int(last.logits[0, -1, :].argmax())

        # (c) the same experiment in the other order: start token, then the token it predicts
        start = torch.tensor([[model.config.text_config.decoder_start_token_id]])
        start_embeds = lm.get_input_embeddings()(start)
        step1 = lm(encoder_outputs=BaseModelOutput(last_hidden_state=enc), attention_mask=mask,
                   decoder_inputs_embeds=start_embeds, use_cache=True, return_dict=True)
        step1_tok = int(step1.logits[0, -1, :].argmax())

        # (d) the prompt followed by the start token: what does the model predict next?
        both = torch.cat([prompt_embeds, start_embeds], dim=1)
        both_mask = torch.ones(both.shape[:2], dtype=torch.float32)
        appended = lm(encoder_outputs=BaseModelOutput(last_hidden_state=enc), attention_mask=mask,
                      decoder_inputs_embeds=both, use_cache=True, return_dict=True)
        appended_tok = int(appended.logits[0, -1, :].argmax())

    out["prompt_only_last_row"] = full_tok
    out["last_prompt_position_only"] = last_tok
    out["start_token_only"] = step1_tok
    out["prompt_plus_start_token"] = appended_tok
    out["old_and_new_agree"] = full_tok == last_tok
    out["start_matches_prompt"] = step1_tok == full_tok

    print(json.dumps(out, indent=2))
    verdict = (
        "SAFE: prefill on the last prompt position gives the same token as the whole prompt"
        if out["old_and_new_agree"]
        else "UNSAFE: they disagree; revert the one-position prefill before exporting"
    )
    print(verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
