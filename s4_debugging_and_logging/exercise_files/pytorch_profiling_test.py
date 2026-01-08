import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet34().to(device)
inputs = torch.randn(5, 3, 224, 224).to(device)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))

prof.export_chrome_trace("trace.json")

from torch.profiler import profile, tensorboard_trace_handler
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, on_trace_ready=tensorboard_trace_handler("./log/resnet34")) as prof:
    for i in range(10):
        model(inputs)