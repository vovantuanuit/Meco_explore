import torch

from . import indicator
from ..p_utils import get_layer_metric_array_dss
import torch.nn as nn

@indicator('dss', bn=False, mode='param')
def compute_dss_per_weight(net, inputs, targets, mode, split_data=1, loss_fn=None):

    device = inputs.device

    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    signs = linearize(net)

    net.zero_grad()
    input_dim = list(inputs[0,:].shape)
    inputs = torch.ones([1] + input_dim).float().to(device)
    output = net.forward(inputs)
    torch.sum(output).backward()

    def dss(layer):
        if layer._get_name() == 'PatchembedSuper':
            if layer.sampled_weight.grad is not None:
                return torch.abs(layer.sampled_weight.grad * layer.sampled_weight)
            else:
                return torch.zeros_like(layer.sampled_weight)
        if isinstance(layer, nn.Linear) and 'qkv' in layer._get_name() and layer.samples or isinstance(layer,
                                                                                                       nn.Linear) and layer.out_features == layer.in_features and layer.samples:
            if layer.samples['weight'].grad is not None:
                return torch.abs(
                    torch.norm(layer.samples['weight'].grad, 'nuc') * torch.norm(layer.samples['weight'], 'nuc'))
            else:
                return torch.zeros_like(layer.samples['weight'])
        if isinstance(layer,
                      nn.Linear) and 'qkv' not in layer._get_name() and layer.out_features != layer.in_features and layer.out_features != 1000 and layer.samples:
            if layer.samples['weight'].grad is not None:
                return torch.abs(layer.samples['weight'].grad * layer.samples['weight'])
            else:
                return torch.zeros_like(layer.samples['weight'])
        elif isinstance(layer, torch.nn.Linear) and layer.out_features == 1000:
            if layer.weight.grad is not None:
                return torch.abs(layer.weight.grad * layer.weight)
            else:
                return torch.zeros_like(layer.weight)
        else:
            return torch.tensor(0).to(device)
    grads_abs = get_layer_metric_array_dss(net, dss, mode)

    nonlinearize(net, signs)

    return grads_abs

# [Xinda], fork from compute_dss_per_weight by the original dss code
@indicator('AutoProxA', bn=False, mode='param')
def compute_AutoProxA_per_weight(net, inputs, targets, mode, split_data=1, loss_fn=None):

    device = inputs.device

    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    signs = linearize(net)

    net.zero_grad()
    input_dim = list(inputs[0,:].shape)
    inputs = torch.ones([1] + input_dim).float().to(device)
    output = net.forward(inputs)
    torch.sum(output).backward()

    def AutoProxA(layer):
        if layer._get_name() == 'PatchembedSuper':  # [Xinda] ‘PatchembedSuper’ is a class in model/module/embedding_super.py, it uses a conv2D to convert the image into embeddings
            if layer.sampled_weight.grad is not None:
                return torch.abs(layer.sampled_weight.grad * layer.sampled_weight)
            else:
                return torch.zeros_like(layer.sampled_weight)  # [Xinda] this subsection is useless, because when 'mode' = 'param', get_layer_metric_array_dss only compute the proxy for nn.Linear, which does not exist in PatchembedSuper.
        if isinstance(layer, nn.Linear) and 'qkv' in layer._get_name() and layer.samples or isinstance(layer,
                                                                                                       nn.Linear) and layer.out_features == layer.in_features and layer.samples:  # [Xinda] if 'qkv' exists in the name of the layer, it is considered 'MSA' layer, the proxy is computed based on Eq.(4)
            if layer.samples['weight'].grad is not None:
                return torch.norm(layer.samples['weight'].grad, p=1)  # [Xinda] L1-norm of the gradient of the weights, see Equation 2 & Figure 7 in AAAI24 paper
            else:
                return torch.zeros_like(layer.samples['weight'])
        if isinstance(layer,
                      nn.Linear) and 'qkv' not in layer._get_name() and layer.out_features != layer.in_features and layer.out_features != 1000 and layer.samples:  # [Xinda] if 'qkv' does not exist in the name of the layer, it is considered 'MLP' layers, the proxy is computed based on Eq.(5)
            if layer.samples['weight'].grad is not None:
                return torch.sum(torch.sigmoid(layer.samples['weight'].grad))/(torch.numel(layer.samples['weight'].grad) + 1e-9)
            else:
                return torch.zeros_like(layer.samples['weight'])
        elif isinstance(layer, torch.nn.Linear) and layer.out_features == 1000:  # [Xinda] for some other MLP layers, the proxy is also computed based on Eq.(5), the difference is only about "layer.samples['weight'].grad" and "layer.weight.grad"
            if layer.weight.grad is not None:
                return torch.sum(torch.sigmoid(layer.weight.grad))/(torch.numel(layer.weight.grad) + 1e-9)
            else:
                return torch.zeros_like(layer.weight)
        else:  # otherwise, it does not contribute to the proxy
            return torch.tensor(0).to(device)
    grads_abs = get_layer_metric_array_dss(net, AutoProxA, mode)  # [Xinda] I kept the name 'get_layer_metric_array_dss' because it is simpler

    nonlinearize(net, signs)

    return grads_abs


