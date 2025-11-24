#!/usr/bin/env python3
"""Script to pre-download the GLM model."""

from transformers import AutoTokenizer, AutoModelForCausalLM

print("Downloading GLM model... This may take a while.")
model_name = "THUDM/chatglm3-6b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True, device_map="auto")
print("Model downloaded successfully!")
