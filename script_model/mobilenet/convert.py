import torch
from torchvision import models
from torch.utils.mobile_optimizer import optimize_for_mobile
model = models.mobilenet_v2(pretrained=True)
scripted_m=torch.jit.script(model)
opt_m=optimize_for_mobile(scripted_m,backend='vulkan')
torch.jit.save(opt_m, "mobilenet2-vulkan.pt")
### for CPU use
#opt_m=optimize_for_mobile(scipted_m)
#opt_m._save_for_lite_interpreter("mobilenet_v2_vulkan.ptl")
