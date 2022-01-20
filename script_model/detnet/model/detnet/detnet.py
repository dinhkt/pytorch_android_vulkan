'''
detnet  based on PyTorch
this is modified from https://github.com/lingtengqiu/Minimal-Hand
'''
import sys

import torch

sys.path.append("./")
from torch import nn
from einops import rearrange, repeat
from model.helper import resnet50, conv3x3
import numpy as np


# my modification
def get_pose_tile_torch(N : int):
    # pos_tile = np.expand_dims(
    #     np.stack(
    #         [
    #             np.tile(np.linspace(-1, 1, 32).reshape([1, 32]), [32, 1]),
    #             np.tile(np.linspace(-1, 1, 32).reshape([32, 1]), [1, 32])
    #         ], -1
    #     ), 0
    # )
    pos_tile = torch.stack(
            [
                torch.tile(torch.linspace(-1, 1, 32).reshape(1, 32), (32, 1)),
                torch.tile(torch.linspace(-1, 1, 32).reshape(32, 1), (1, 32))
            ], -1
        )
    pos_tile = pos_tile[np.newaxis, :, :, :]
    pos_tile = torch.tile(pos_tile, (N, 1, 1, 1))
    retv = pos_tile.float()
    # return rearrange(retv, 'b h w c -> b c h w')
    return retv.permute(0,3,1,2)


class net_2d(nn.Module):
    def __init__(self, input_features, output_features, stride, joints=21):
        super().__init__()
        self.project = nn.Sequential(conv3x3(input_features, output_features, stride), nn.BatchNorm2d(output_features),
                                     nn.ReLU())

        self.prediction = nn.Conv2d(output_features, joints, 1, 1, 0)

    def forward(self, x):
        x = self.project(x)
        x = self.prediction(x).sigmoid()
        return x


class net_3d(nn.Module):
    def __init__(self, input_features, output_features, stride, joints=21, need_norm=False):
        super().__init__()
        self.need_norm = need_norm
        self.project = nn.Sequential(conv3x3(input_features, output_features, stride), nn.BatchNorm2d(output_features),
                                     nn.ReLU())
        self.prediction = nn.Conv2d(output_features, joints * 3, 1, 1, 0)

    def forward(self, x):
        x = self.prediction(self.project(x))

        # dmap = rearrange(x, 'b (j l) h w -> b j l h w', l=3)
        dmap = x.reshape(x.shape[0], x.shape[1]//3, 3, x.shape[2], x.shape[3])

        return dmap


class detnet_jit(torch.jit.ScriptModule):
    def __init__(self, stacks=1):
        super().__init__()
        self.resnet50 = resnet50()

        self.hmap_0 = net_2d(258, 256, 1)
        self.dmap_0 = net_3d(279, 256, 1)
        self.lmap_0 = net_3d(342, 256, 1)
        self.stacks = stacks
    @torch.jit.script_method
    def forward(self, x):
        features = self.resnet50(x)

        device = x.device
        pos_tile = get_pose_tile_torch(features.shape[0]).to(device)

        x = torch.cat([features, pos_tile], dim=1)

        hmaps = []
        dmaps = []
        lmaps = []

        for _ in range(self.stacks):
            heat_map = self.hmap_0(x)
            hmaps.append(heat_map)
            x = torch.cat([x, heat_map], dim=1)

            dmap = self.dmap_0(x)
            dmaps.append(dmap)

            # x = torch.cat([x, rearrange(dmap, 'b j l h w -> b (j l) h w')], dim=1)
            x = torch.cat([x, 
                dmap.reshape(dmap.shape[0],dmap.shape[1]*dmap.shape[2],
                    dmap.shape[3],dmap.shape[4])], dim=1)

            lmap = self.lmap_0(x)
            lmaps.append(lmap)
        hmap, dmap, lmap = hmaps[-1], dmaps[-1], lmaps[-1]

        x = hmap.view((hmap.shape[0],hmap.shape[1],hmap.shape[2]*hmap.shape[3])).max(dim=2)[1]%32
        y = hmap.view((hmap.shape[0],hmap.shape[1],hmap.shape[2]*hmap.shape[3])).max(dim=2)[1]//32
        hmap_ = torch.cat((x.unsqueeze(1),y.unsqueeze(1)), dim=1)
        hmap_ = hmap_.permute(0,2,1)        
        
        uv, argmax = self.map_to_uv(hmap)

        delta = self.dmap_to_delta(dmap, argmax)
        xyz = self.lmap_to_xyz(lmap, argmax)

        det_result = {
            "h_map": hmap,
            "d_map": dmap,
            "l_map": lmap,
            "delta": delta,
            "xyz": xyz,
            "uv": uv,
            "hmap_":hmap_
        }

        return det_result

    # @property
    # def pos(self):
    #     return self.__pos_tile

    @staticmethod
    def map_to_uv(hmap):
        b, j, h, w = hmap.shape
        # hmap = rearrange(hmap, 'b j h w -> b j (h w)')
        hmap = hmap.reshape(hmap.shape[0], hmap.shape[1], hmap.shape[2]*hmap.shape[3])
        argmax = torch.argmax(hmap, -1, keepdim=True)
        u = argmax // w
        v = argmax % w
        uv = torch.cat([u, v], dim=-1)

        return uv, argmax

    def dmap_to_delta(self, dmap, argmax):
        return self.lmap_to_xyz(dmap, argmax)

    def lmap_to_xyz(self, lmap, argmax):
        # lmap = rearrange(lmap, 'b j l h w -> b j (h w) l')
        lshape = lmap.shape
        lmap = lmap.permute(0, 1, 3, 4, 2).reshape(lshape[0],lshape[1],lshape[3]*lshape[4],lshape[2])
        # index = repeat(argmax, 'b j i -> b j i c', c=3)
        index = argmax[:,:,:,np.newaxis].repeat(1,1,1,3)
        xyz = torch.gather(lmap, dim=2, index=index).squeeze(2)
        return xyz


if __name__ == '__main__':
    import time

    mydet = detnet_jit()
    img_crop = torch.randn(10, 3, 128, 128)
    res = mydet(img_crop)
    st = time.time()
    for i in range(1):
        img_crop = torch.randn(10, 3, 128, 128)
        res = mydet(img_crop)
    print((time.time()-st)/10)
    hmap = res["h_map"]
    dmap = res["d_map"]
    lmap = res["l_map"]
    delta = res["delta"]
    xyz = res["xyz"]
    uv = res["uv"]
    hmap_= res["hmap_"]
    print("hmap.shape=", hmap.shape)
    print("dmap.shape=", dmap.shape)
    print("lmap.shape=", lmap.shape)
    print("delta.shape=", delta.shape)
    print("xyz.shape=", xyz.shape)
    print("uv.shape=", uv.shape)
    print("hmap_.shape=", hmap_.shape)
    