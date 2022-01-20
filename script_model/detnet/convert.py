import torch
from model.detnet import detnet_jit
from torch.utils.mobile_optimizer import optimize_for_mobile
model = detnet_jit()
checkpoint=torch.load('ckp_detnet_106.pth',map_location='cpu')
model_state=model.state_dict()
state={}
for k,v in checkpoint.items():
    if k in model_state:
        state[k]=v
    else:
        print(k,'is not in model')
model_state.update(state)
model.load_state_dict(model_state)
model.eval()
print("done loading model")

#sample_inp=torch.rand(1,3,128,128)
#scripted_m=torch.jit.trace(model,sample_inp,strict=False)
scripted_m=torch.jit.script(model)
opt_m=optimize_for_mobile(scripted_m,backend='vulkan')
torch.jit.save(opt_m,"detnet_vulkan.pt")
### For cpu use
#opt_m=optimize_for_mobile(scripted_m)
#opt_m._save_for_lite_interpreter("detnet_script_hmap.ptl")
