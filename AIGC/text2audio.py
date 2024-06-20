from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import torch

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
if torch.backends.mps.is_available():
    device = "mps"
if torch.xpu.is_available():
    device = "xpu"
torch_dtype = torch.float16 if device != "cpu" else torch.float32

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1").to(device, dtype=torch_dtype)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")

prompt = "Good morning, sir. How can I help you?"
description = "A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very confined sounding environment with clear audio quality. She speaks very fast."

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids).to(torch.float32)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("generated output/parler_tts_out2.wav", audio_arr, model.config.sampling_rate)