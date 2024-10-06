import torch
import time

resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')
zero_time = time.time()
resnet50.eval().to(device)

uris = [
    'http://images.cocodataset.org/test-stuff2017/000000024309.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000028117.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000006149.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000004954.jpg',
]

batch = torch.cat(
    [utils.prepare_input_from_uri(uri) for uri in uris]
).to(device)

start_time = time.time()
total_samples = 0
with torch.no_grad():
    output = torch.nn.functional.softmax(resnet50(batch), dim=1)
    
    total_samples += batch.size(0)
end_time = time.time()

if device.type == 'cuda':
    torch.cuda.synchronize() # Make sure all operations are finished

elapsed_time = end_time - start_time
print(f"Total samples : {total_samples}")
print(f"Throughput: {total_samples/elapsed_time:.6f} samples/second")
print(f"Latency: {end_time - zero_time}")

# results = utils.pick_n_best(predictions=output, n=5)
# print(results)
