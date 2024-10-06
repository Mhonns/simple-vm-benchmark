import whisper
import torch
import time

total_samples = 1
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load model to the gpu
load_start_time = time.time()
model = whisper.load_model("base", device="cuda")
if device.type == 'cuda':
    torch.cuda.synchronize()
load_end_time = time.time()

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("audio.mp3")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
# print(f"Detected language: {max(probs, key=probs.get)}")

# Decode the audio
options = whisper.DecodingOptions()
inference_start_time = time.time()
result = whisper.decode(model, mel, options)
if device.type == 'cuda':
    torch.cuda.synchronize() # Make sure all operations are finished
inference_stop_time = time.time()

# Getting the result
inference_time = inference_stop_time - inference_start_time
throughput = total_samples / inference_time

print(f"Total samples : {total_samples}")
print(f"Throughput: {throughput:.2f} samples/second")
print(f"Load model latency: {load_end_time - load_start_time}")
print(f"Inference latency: {inference_time}")

# print the recognized text
print(result.text)


