import whisper
import torch
import time

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

start_time = time.time()
model = whisper.load_model("base", device="cuda")
end_time = time.time()
time_taken = end_time - start_time
print(f"Time taken to move the model to {device}: {time_taken:.4f} seconds")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("audio.mp3")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
# print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
start_time = time.time()
result = whisper.decode(model, mel, options)
if device.type == 'cuda':
    torch.cuda.synchronize() # Make sure all operations are finished
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Throughput: {(1/elapsed_time):.6f} samples/second")
print(f"Latency: {time_taken + elapsed_time}")
# print the recognized text
print(result.text)


