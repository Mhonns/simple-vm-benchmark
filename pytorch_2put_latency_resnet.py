import torch
import time

resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

# Load model to the gpu
load_start_time = time.time()
resnet50.eval().to(device)
if device.type == 'cuda':
    torch.cuda.synchronize() # Make sure all operations are finished
load_end_time = time.time()

# Load samples to the gpu
uris = [
    'http://images.cocodataset.org/test-stuff2017/000000024309.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000028117.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000006149.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000004954.jpg',
]
batch = torch.cat(
    [utils.prepare_input_from_uri(uri) for uri in uris]
).to(device)

# Start inferencing
total_samples = 0
inference_start_time = time.time()
with torch.no_grad():
    output = torch.nn.functional.softmax(resnet50(batch), dim=1)
    total_samples += batch.size(0)
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

results = utils.pick_n_best(predictions=output, n=5)
print(results)