"""Vendored SLS + XLS-R architecture from Yash-Sukhdeve/XLS-R-SLS-Deepfake-Detection.

Reference: Zhang et al., ACM Multimedia 2024.
Requires fairseq + xlsr2_300m.pt base checkpoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSLModel(nn.Module):
    def __init__(self, device, cp_path="xlsr2_300m.pt"):
        super().__init__()
        import fairseq

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device = device
        self.out_dim = 1024

    def extract_feat(self, input_data):
        if (
            next(self.model.parameters()).device != input_data.device
            or next(self.model.parameters()).dtype != input_data.dtype
        ):
            self.model.to(input_data.device, dtype=input_data.dtype)
        self.model.train()

        input_tmp = input_data[:, :, 0] if input_data.ndim == 3 else input_data
        out = self.model(input_tmp, mask=False, features_only=True)
        return out["x"], out["layer_results"]


def get_atten_f(layer_result):
    poollayer_result = []
    fullf = []
    for layer in layer_result:
        layery = layer[0].transpose(0, 1).transpose(1, 2)
        layery = F.adaptive_avg_pool1d(layery, 1)
        layery = layery.transpose(1, 2)
        poollayer_result.append(layery)

        x = layer[0].transpose(0, 1)
        x = x.view(x.size(0), -1, x.size(1), x.size(2))
        fullf.append(x)

    layery = torch.cat(poollayer_result, dim=1)
    fullfeature = torch.cat(fullf, dim=1)
    return layery, fullfeature


class SLSDeepfakeModel(nn.Module):
    def __init__(self, device, xlsr_path="xlsr2_300m.pt"):
        super().__init__()
        self.device = device
        self.ssl_model = SSLModel(device, cp_path=xlsr_path)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()
        self.fc1 = nn.Linear(22847, 1024)
        self.fc3 = nn.Linear(1024, 2)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        _, layer_result = self.ssl_model.extract_feat(x.squeeze(-1))
        y0, fullfeature = get_atten_f(layer_result)
        y0 = self.sig(self.fc0(y0))
        y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2], -1)
        fullfeature = fullfeature * y0
        fullfeature = torch.sum(fullfeature, 1).unsqueeze(dim=1)
        x = self.first_bn(fullfeature)
        x = self.selu(x)
        x = F.max_pool2d(x, (3, 3))
        x = torch.flatten(x, 1)
        x = self.selu(self.fc1(x))
        x = self.selu(self.fc3(x))
        return self.logsoftmax(x)
