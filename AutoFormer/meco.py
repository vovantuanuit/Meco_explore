# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
import copy
import time

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import numpy as np
import torch
from torch import nn
lossfunc = nn.CrossEntropyLoss().cuda()
import torch.nn.functional as F 
# from . import measure

import pickle

def get_score_zico(net, x, target, device, split_data):
    result_list = []
    
    outputs = net.forward(x)
    loss = lossfunc(outputs, target)
    loss.backward()
    def forward_hook(module, data_input, data_output):
        # fea_or = data_output[0]
        fea = data_output[0].detach()
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # grad = fea_or.grad
            
            grad = module.weight.grad.data.cpu().reshape( -1).numpy()
            nsr_std = np.std(grad, axis=0)
            # print(nsr_std)
            # nonzero_idx = np.nonzero(nsr_std)[0]
            nsr_mean_abs = np.mean(np.abs(grad), axis=0)
            tmpsum = np.sum(nsr_mean_abs/nsr_std)
            zico = np.log(tmpsum)
        else:
            zico = 1.0
        
        # fea = fea*grad
        fea = fea.reshape(fea.shape[0], -1)
        
        corr = torch.corrcoef(fea)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.min(torch.real(values))*zico
        result_list.append(result)

    for name, modules in net.named_modules():
        # print(modules)
        # print('----------------------------------------------------------')
        modules.register_forward_hook(forward_hook)



    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        y = net(x[st:en])
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item()


def get_score_Meco(net, x, target, device, split_data):
    result_list = []
    
    def forward_hook(module, data_input, data_output):
        # print('---day')
        fea = data_output[0].detach()
        fea = fea.reshape(fea.shape[0], -1)
        corr = torch.corrcoef(fea)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.min(torch.real(values))
        result_list.append(result)

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)



    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        y = net(x[st:en])
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item()

def get_score_Meco_result(net, x, target, device, split_data):
    result_list = []
    meco_layers = []
    eigen_layers = []
    layer_shape_C = []
    def forward_hook(module, data_input, data_output):
        # print('---day')
        fea = data_output[0].detach()
        fea = fea.reshape(fea.shape[0], -1)
        corr = torch.corrcoef(fea)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.min(torch.real(values))
        result_list.append(result)
        meco_layers.append(result.item())
        eigen_layers.append(torch.real(values).tolist())
        layer_shape_C.append((fea.shape[0],fea.shape[0]))

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)



    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        y = net(x[st:en])
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item(),meco_layers,eigen_layers,layer_shape_C
def cout_cor(corr,x):
    # print(corr.shape)
    if x == 1:
        corr = corr.detach().cpu().numpy()
        print(corr.shape)
        print(np.argwhere(corr==1.).shape)
        index = np.argwhere(corr==1.).shape[0]
        
        num_ = index - corr.shape[0]
        # print(x)
        # print('index',num_)
        # print('shape',corr.shape[0])
    else:
        # print(x)
        print(corr.shape)
        
        corr = corr.detach().cpu().numpy()
        print(np.argwhere(corr>x).shape)
        index = np.argwhere(corr>x).shape[0]
        num_ = index - corr.shape[0]
        # print('index',num_)
        # print('shape',corr.shape[0])
    # print(num_)
    return num_
    

def get_score_Meco_result_ident_zero(net, x, target, device, split_data):
    result_list = []
    meco_layers = []
    eigen_layers = []
    layer_shape_C = []
    identical_percent_layers = []
    zero_percent_laeyers = []
    def forward_hook(module, data_input, data_output):
        # print('---day')
        fea = data_output[0] #.detach()
        # print(fea.shape)
        # print((fea.sum(dim=0) == 0))
        # zero_matrices = (fea.sum(dim=0) == 0).sum(dim=(1, 2))
        # num_zero_matrices = 0
        # for i in range(fea.shape[0]):
        #     # print(fea[i].shape)
        #     if torch.sum(fea[i]).item() == 0:
        #         num_zero_matrices+=1
                # print('o day')
        # print(num_zero_matrices)
        try:
            num_zero_sum_matrices = (fea.sum(dim=[1, 2]) == 0).sum()
        except:
            num_zero_sum_matrices= 0
        
        percen_zero = num_zero_sum_matrices/fea.shape[0]

        try:
            percen_zero = percen_zero.item()
        except:
            percen_zero = percen_zero
        fea = fea.reshape(fea.shape[0], -1)

        
        corr = torch.corrcoef(fea)
        # print('shape: ',corr.shape)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        corr = corr.float().cuda()
        # try:
        values = torch.linalg.eig(corr)[0]
        # except:
            # pass
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.min(torch.real(values))
        # percent_layer_corr = {}
        # for i in [0.5,0.6,0.7,0.8,0.9,1]:
        #     corr_num_percent = cout_cor(corr,i)/corr.shape[0]
        #     print('corr: ',corr_num_percent)
        #     percent_layer_corr[i] = corr_num_percent

        result_list.append(result)
        meco_layers.append(result.item())
        eigen_layers.append(torch.real(values).tolist())
        layer_shape_C.append((fea.shape[0],fea.shape[1]))
        zero_percent_laeyers.append(percen_zero)
        # identical_percent_layers.append(percent_layer_corr)

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)



    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        y = net(x[st:en].cuda())
        del y
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item(),meco_layers,eigen_layers,layer_shape_C,zero_percent_laeyers #,identical_percent_layers


def get_score_Meco_result_heatmap(net, x, target, device, split_data):
    result_list = []
    meco_layers = []
    heamaps = []
    def forward_hook(module, data_input, data_output):
        # print('---day')
        fea = data_output[0].detach()
        fea = fea.reshape(fea.shape[0], -1)
        
        corr = torch.corrcoef(fea)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.min(torch.real(values))
        result_list.append(result)
        meco_layers.append(result.item())
        heamaps.append(corr)

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)



    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        y = net(x[st:en])
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item(),meco_layers,heamaps

def get_score_Meco_input_random_result(net, x, target, device, split_data):
    result_list = []
    meco_layers = []
    eigen_layers = []
    def forward_hook(module, data_input, data_output):
        # print('---day')
        fea = data_output[0].detach()
        fea = fea.reshape(fea.shape[0], -1)
        corr = torch.corrcoef(fea)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.min(torch.real(values))
        result_list.append(result)
        meco_layers.append(result.item())
        eigen_layers.append(torch.real(values).tolist())

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)


    

# Generate a tensor with random values from a standard Gaussian distribution
    x = torch.randn(size=x.shape).to('cuda')
    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        y = net(x[st:en])
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item(),meco_layers,eigen_layers

def get_score_Meo_8x8_opt(net, x, target, device, split_data):
    result_list = []
    temp_list = []
    def forward_hook(module, data_input, data_output):
        # print('---day')
        fea = data_output[0].detach()
        fea = fea.reshape(fea.shape[0], -1)
        random_indices_8_a = torch.randperm(fea.shape[0])[:8]  # Get 8 random indices
        random_tensor_8_a_fea = fea[random_indices_8_a]
        corr = torch.corrcoef(random_tensor_8_a_fea)
        # print(corr.shape)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.min(torch.real(values))
        result_list.append(result)
        temp_list.append(result.item())

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)



    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        y = net(x[st:en])
    
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    # result_list.clear()
    return v.item(),temp_list

def get_score_Meco_16x16_opt_weight_result(net, x, target, device, split_data):
    result_list = []
    meco_layers = []
    eigen_layers = []
    layer_shape_C = []
    def forward_hook(module, data_input, data_output):
        # print('---day')
        fea = data_output[0].detach()
        fea = fea.reshape(fea.shape[0], -1)
        random_indices_8_a = torch.randperm(fea.shape[0])[:16]  # Get 8 random indices
        random_tensor_8_a_fea = fea[random_indices_8_a]
        corr = torch.corrcoef(random_tensor_8_a_fea)
        # print(corr.shape)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = (fea.shape[0]/16)*torch.min(torch.real(values))
        result_list.append(result)
        meco_layers.append(result.item())
        eigen_layers.append(torch.real(values).tolist())
        layer_shape_C.append((fea.shape[0],fea.shape[0]))
        # print(layer_shape_C)

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)



    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        y = net(x[st:en])
    
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item(),meco_layers,eigen_layers,layer_shape_C

def get_score_Meco_32x32_opt_weight_result(net, x, target, device, split_data):
    result_list = []
    meco_layers = []
    eigen_layers = []
    layer_shape_C = []
    def forward_hook(module, data_input, data_output):
        # print('---day')
        fea = data_output[0].detach()
        fea = fea.reshape(fea.shape[0], -1)
        random_indices_8_a = torch.randperm(fea.shape[0])[:32]  # Get 8 random indices
        random_tensor_8_a_fea = fea[random_indices_8_a]
        corr = torch.corrcoef(random_tensor_8_a_fea)
        # print(corr.shape)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = (fea.shape[0]/32)*torch.min(torch.real(values))
        result_list.append(result)
        meco_layers.append(result.item())
        eigen_layers.append(torch.real(values).tolist())
        layer_shape_C.append((fea.shape[0],fea.shape[0]))
        # print(layer_shape_C)

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)



    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        y = net(x[st:en])
    
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item(),meco_layers,eigen_layers,layer_shape_C

def get_score_Meco_8x8_opt_weight_result(net, x, target, device, split_data):
    result_list = []
    meco_layers = []
    eigen_layers = []
    layer_shape_C = []
    def forward_hook(module, data_input, data_output):
        # print('---day')
        fea = data_output[0].detach()
        fea = fea.reshape(fea.shape[0], -1)
        random_indices_8_a = torch.randperm(fea.shape[0])[:8]  # Get 8 random indices
        random_tensor_8_a_fea = fea[random_indices_8_a]
        corr = torch.corrcoef(random_tensor_8_a_fea)
        # print(corr.shape)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = (fea.shape[0]/8)*torch.min(torch.real(values))
        result_list.append(result)
        meco_layers.append(result.item())
        eigen_layers.append(torch.real(values).tolist())
        layer_shape_C.append((fea.shape[0],fea.shape[0]))
        # print(layer_shape_C)

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)



    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        y = net(x[st:en])
    
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item(),meco_layers,eigen_layers,layer_shape_C

def get_score_Meco_8x8_opt_weight_result(net, x, target, device, split_data):
    result_list = []
    meco_layers = []
    eigen_layers = []
    layer_shape_C = []
    def forward_hook(module, data_input, data_output):
        # print('---day')
        fea = data_output[0].detach()
        fea = fea.reshape(fea.shape[0], -1)
        random_indices_8_a = torch.randperm(fea.shape[0])[:8]  # Get 8 random indices
        random_tensor_8_a_fea = fea[random_indices_8_a]
        corr = torch.corrcoef(random_tensor_8_a_fea)
        # print(corr.shape)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = (fea.shape[0]/8)*torch.min(torch.real(values))
        result_list.append(result)
        meco_layers.append(result.item())
        eigen_layers.append(torch.real(values).tolist())
        layer_shape_C.append((fea.shape[0],fea.shape[1]))
        # print(layer_shape_C)

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)



    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        y = net(x[st:en])
    
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item(),meco_layers,eigen_layers,layer_shape_C

def get_score_Meco_revised_result_matmul(net, x, target, device, split_data,m):
    result_list = []
    meco_layers = []
    eigen_layers = []
    layer_shape_C = []
    def forward_hook(module, data_input, data_output):
        # print('---day')
        fea = data_output[0].detach()
        fea = fea.reshape(fea.shape[0], -1)
        # random_indices_8_a = torch.randperm(fea.shape[0])[:8]  # Get 8 random indices
        # random_tensor_8_a_fea = fea[random_indices_8_a]
        fea_centered = fea - torch.mean(fea,dim=1,keepdim=True)
        l2_norm = torch.clamp(torch.norm(fea_centered,p=2,dim=1),min=1e-8)
        feanorm = fea_centered/l2_norm.unsqueeze(-1)
        gram = torch.matmul(feanorm,feanorm.t())
        gram[torch.isnan(gram)] = 0

        simi_sums = torch.sum(torch.abs(gram),dim=1)
        _,fea_index = torch.sort(simi_sums,descending=False)
        fea_index = fea_index[:m]
        gram = gram[fea_index][:,fea_index]

        
        values = torch.linalg.eig(gram)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.min(torch.real(values))
        result_list.append(result)
        meco_layers.append(result.item())
        eigen_layers.append(torch.real(values).tolist())
        layer_shape_C.append((fea.shape[0],fea.shape[1]))
        # print(layer_shape_C)

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)



    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        y = net(x[st:en])
    
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item(),meco_layers,eigen_layers,layer_shape_C

def get_lowest_values(input_matrix, n):
    # Ensure that n is less than C
    # print(input_matrix.shape)
    # assert n < input_matrix.size(0) and n < input_matrix.size(1), "n should be less than C"
    
    # Reshape the input matrix to an nxnx...xn tensor, where each dimension is 2
    unfolded_matrix = input_matrix.unfold(0, n, n).unfold(1, n, n)
    
    # Reshape to flatten the last two dimensions
    flattened_matrix = unfolded_matrix.reshape(-1, n, n)
    
    # Take the minimum value along the first dimension
    result_matrix, _ = flattened_matrix.min(dim=0)
    
    return result_matrix

# def get_lowest_values(input_matrix, n):
#     # Ensure that n is less than C
#     # assert n < input_matrix.size(0) and n < input_matrix.size(1), "n should be less than C"

#     # Unfold the input matrix into overlapping blocks
#     unfolded_matrix = input_matrix.unfold(0, n, 1).unfold(1, n, 1)

#     # Reshape to a 3D tensor with shape (num_blocks, n, n)
#     unfolded_matrix = unfolded_matrix.contiguous().view(-1, n, n)

#     # Take the minimum along the first dimension (num_blocks)
#     result_matrix, _ = unfolded_matrix.min(dim=0)

#     return result_matrix


def get_score_Meco_revised_updated(net, x, target, device, split_data,m):
    result_list = []
    meco_layers = []
    eigen_layers = []
    layer_shape_C = []
    def forward_hook(module, data_input, data_output):
        # print('---day')
        fea = data_output[0].detach()
        fea = fea.reshape(fea.shape[0], -1)

        n = fea.shape[0]
        fea_centered = fea - torch.mean(fea, dim=1, keepdim=True)
        l2_norm = torch.clamp(torch.norm(fea_centered, p=2, dim=1), min=1e-8)
        feanorm = fea_centered / l2_norm.unsqueeze(-1)
        gram = torch.matmul(feanorm, feanorm.t())
        gram[torch.isnan(gram)] = 0
        gram_org = gram.clone()

        # from Tuyen's scheme
        simi_sums = torch.sum(torch.abs(gram), dim=1)
        # simi_sums_dict[new_name] = simi_sums.tolist()

        fea_index = set()
        mask = torch.triu(torch.ones(n, n), diagonal=0).bool()
        gram_masked = gram.clone()
        gram_masked[mask] = 0
        count = 0
        curselect = torch.zeros(gram_masked.shape[1], dtype=bool)

        while len(fea_index) < m:
            curselect[list(fea_index)] = True
            if torch.sum(gram_masked[:, ~curselect]) == 0:  # if all non-selected columns are zeros, the remaining ones are all zero-features
                # print('Current selections are {', fea_index ,'}, and the remainings are all zero-features')
                rem_s = set(range(0, n))
                rem_s -= fea_index
                rem_s = set(list(rem_s)[:m - len(fea_index)])
            else:
                # print('Count: ', count)
                if len(fea_index) < m - 1:
                    max_value, max_index = torch.max(torch.abs(gram_masked).view(-1), 0)
                    max_row, max_col = divmod(max_index.item(), n)
                    # print('The current masked gram is: ', gram_masked)
                    # print('max_row:', max_row)
                    # print('max_col:', max_col)
                    # print('The element to be removed is:', gram_masked[max_row, max_col])
                    gram_masked[max_row, max_col] = 0
                    rem_s = {max_row, max_col}
                else:
	       # if there is only one feature to be selected, choose the feature that is most similar to the ones already selected (i.e., the one with the highest summed similarities to the selected features)
                    mask_final = torch.zeros(gram_masked.shape[1], dtype=bool)
                    mask_final[list(fea_index)] = True
                    gram[:, ~mask_final] = 0
                    # print('Only check the gram for selected features: ', gram)
                    gram[mask_final, :] = 0
                    # print('Only check the gram for candidate rows: ', gram)
                    # print('The sums are: ', torch.sum(torch.abs(gram), dim=1))
                    _, max_index = torch.max(torch.sum(torch.abs(gram), dim=1), 0)
                    # print('max_index:', max_index.item())
                    rem_s = {max_index.item()}
            count += 1
            # print('The set was: ', fea_index)
            fea_index |= rem_s
            # print('After the operation the set is: ', fea_index)
        # print("The elements in the set are:", fea_index)
        fea_index = torch.LongTensor(list(fea_index))
        gram = gram_org[fea_index][:, fea_index]
        values = torch.linalg.eig(gram)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.min(torch.real(values))
        result_list.append(result)
        meco_layers.append(result.item())
        eigen_layers.append(torch.real(values).tolist())
        layer_shape_C.append((fea.shape[0],fea.shape[1]))

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)



    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        y = net(x[st:en])
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item(),meco_layers,eigen_layers,layer_shape_C

def get_score_Meco_revised_result_matmul_oppiste(net, x, target, device, split_data,m):
    result_list = []
    meco_layers = []
    eigen_layers = []
    layer_shape_C = []
    def forward_hook(module, data_input, data_output):
        # print('---day')
        fea = data_output[0].detach()
        fea = fea.reshape(fea.shape[0], -1)
        # random_indices_8_a = torch.randperm(fea.shape[0])[:8]  # Get 8 random indices
        # random_tensor_8_a_fea = fea[random_indices_8_a]
        fea_centered = fea - torch.mean(fea,dim=1,keepdim=True)
        l2_norm = torch.clamp(torch.norm(fea_centered,p=2,dim=1),min=1e-8)
        feanorm = fea_centered/l2_norm.unsqueeze(-1)
        gram = torch.matmul(feanorm,feanorm.t())
        gram[torch.isnan(gram)] = 0

        simi_sums = torch.sum(torch.abs(gram),dim=1)
        _,fea_index = torch.sort(simi_sums,descending=False)
        fea_index = fea_index[-m:]
        gram = gram[fea_index][:,fea_index]

        
        values = torch.linalg.eig(gram)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.min(torch.real(values))
        result_list.append(result)
        meco_layers.append(result.item())
        eigen_layers.append(torch.real(values).tolist())
        layer_shape_C.append((fea.shape[0],fea.shape[0]))
        # print(layer_shape_C)

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)



    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        y = net(x[st:en])
    
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item(),meco_layers,eigen_layers,layer_shape_C



def get_score_Meco_revised_result_matmul_3(net, x, target, device, split_data,m):
    result_list = []
    meco_layers = []
    eigen_layers = []
    layer_shape_C = []
    def forward_hook(module, data_input, data_output):
        # print('---day')
        fea = data_output[0].detach()
        fea = fea.reshape(fea.shape[0], -1)
        n = fea.shape[0]
        fea_centered = fea - torch.mean(fea, dim=1, keepdim=True)
        l2_norm = torch.clamp(torch.norm(fea_centered, p=2, dim=1), min=1e-8)
        feanorm = fea_centered / l2_norm.unsqueeze(-1)
        gram = torch.matmul(feanorm, feanorm.t())
        gram[torch.isnan(gram)] = 0
        gram_org = gram
        simi_sums = torch.sum(torch.abs(gram), dim=1)
        if torch.any(simi_sums == 0):
            result = torch.tensor(0)
            list_eigens = []

        else:
            fea_index = set(range(0, n))
            mask = torch.triu(torch.ones(n, n), diagonal=0).bool()
            gram_masked = gram.clone()
            gram_masked[mask] = 0
            while len(fea_index) > m:
                max_value, max_index = torch.max(torch.abs(gram_masked).view(-1), 0)
                max_row, max_col = divmod(max_index.item(), n)
                if max_row in fea_index or max_col in fea_index:
                    if len(fea_index) > m + 1:
                        rem_s = {max_row, max_col}
                    else:
                        if torch.sum(torch.abs(gram[max_row])) > torch.sum(torch.abs(gram[max_col])):
                            rem_s = {max_row}
                        else:
                            rem_s = {max_col}
                    fea_index -= rem_s
                gram_masked[max_row, max_col] = 0
            fea_index = torch.LongTensor(list(fea_index))
        
            gram = gram[fea_index][:, fea_index]
            values = torch.linalg.eig(gram)[0]
            result = torch.min(torch.real(values))
            list_eigens = torch.real(values).tolist()

        result_list.append(result)
        meco_layers.append(result.item())
        eigen_layers.append(list_eigens)
        layer_shape_C.append((fea.shape[0],fea.shape[1]))
        # print(layer_shape_C)

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)



    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        y = net(x[st:en])
    
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item(),meco_layers,eigen_layers,layer_shape_C

def get_score_Meco_revised_result(net, x, target, device, split_data,m):
    result_list = []
    meco_layers = []
    eigen_layers = []
    layer_shape_C = []
    def forward_hook(module, data_input, data_output):
        # print('---day')
        fea = data_output[0].detach()
        fea = fea.reshape(fea.shape[0], -1)
        # random_indices_8_a = torch.randperm(fea.shape[0])[:8]  # Get 8 random indices
        # random_tensor_8_a_fea = fea[random_indices_8_a]
        corr = torch.corrcoef(fea)
        # print(corr.shape)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0

        simi_sums = torch.sum(torch.abs(corr),dim=1)
        _, fea_index = torch.sort(simi_sums,descending=False)
        fea_index = fea_index[:m]
        corr = corr[fea_index][:,fea_index]
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.min(torch.real(values))
        result_list.append(result)
        meco_layers.append(result.item())
        eigen_layers.append(torch.real(values).tolist())
        layer_shape_C.append((fea.shape[0],fea.shape[0]))
        # print(layer_shape_C)

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)



    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        y = net(x[st:en])
    
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item(),meco_layers,eigen_layers,layer_shape_C



def get_score_Meco_grad(net, x, target, device, split_data):
    result_list = []
    for param in net.parameters():
        param.requires_grad = True
    def forward_hook(module, data_input, data_output):
        dydx = torch.autograd.grad(outputs=data_output, inputs=data_input,
            grad_outputs=torch.ones_like(data_output),
            retain_graph=True)[0]

        fea = data_output[0].detach()
        grad = dydx[0].detach()
        # print(grad)
        if fea.shape == grad.shape:
            # print(fea.shape)
            # print(grad.shape)
            # print('vo day nhe')
            # print('orginal:',fea)
            # mix_grad_fea = torch.cat([fea, grad], dim=1)
            mix_grad_fea = torch.cat([fea, grad], dim=0)
            # mix_grad_fea = fea*grad
            # print('mix', mix_grad_fea.shape)
        else:
            mix_grad_fea = fea
        fea_mix = mix_grad_fea.reshape(mix_grad_fea.shape[0], -1)
        corr = torch.corrcoef(fea_mix)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.min(torch.real(values))
        result_list.append(result)
  

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)

    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        
        in_  = x[st:en]
        # print(in_)
        in_ = torch.tensor(in_, requires_grad=True)
        y = net(in_)
        loss = lossfunc(y, target[st:en])
        loss.backward()
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item()

def get_score_Meco_grad_result(net, x, target, device, split_data):
    result_list = []
    meco_layers = []
    eigen_layers = []
    for param in net.parameters():
        param.requires_grad = True
    def forward_hook(module, data_input, data_output):
        dydx = torch.autograd.grad(outputs=data_output, inputs=data_input,
            grad_outputs=torch.ones_like(data_output),
            retain_graph=True)[0]

        fea = data_output[0].detach()
        grad = dydx[0].detach()
        # print(grad)
        if fea.shape == grad.shape:
            # print(fea.shape)
            # print(grad.shape)
            # print('vo day nhe')
            # print('orginal:',fea)
            mix_grad_fea = torch.cat([fea, grad], dim=0)
            # mix_grad_fea = fea*grad
            # print('mix', grad)
        else:
            mix_grad_fea = fea
        fea_mix = mix_grad_fea.reshape(mix_grad_fea.shape[0], -1)
        corr = torch.corrcoef(fea_mix)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.min(torch.real(values))
        result_list.append(result)
        meco_layers.append(result.item())
        eigen_layers.append(torch.real(values).tolist())
  

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)

    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        
        in_  = x[st:en]
        # print(in_)
        in_ = torch.tensor(in_, requires_grad=True)
        y = net(in_)
        loss = lossfunc(y, target[st:en])
        loss.backward()
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item(),meco_layers,eigen_layers


def get_score_Meco_grad_input_random_result(net, x, target, device, split_data):
    result_list = []
    meco_layers = []
    eigen_layers = []
    for param in net.parameters():
        param.requires_grad = True
    def forward_hook(module, data_input, data_output):
        dydx = torch.autograd.grad(outputs=data_output, inputs=data_input,
            grad_outputs=torch.ones_like(data_output),
            retain_graph=True)[0]

        fea = data_output[0].detach()
        grad = dydx[0].detach()
        # print(grad)
        if fea.shape == grad.shape:
            # print(fea.shape)
            # print(grad.shape)
            # print('vo day nhe')
            # print('orginal:',fea)
            mix_grad_fea = torch.cat([fea, grad], dim=0)
            # mix_grad_fea = fea*grad
            # print('mix', grad)
        else:
            mix_grad_fea = fea
        fea_mix = mix_grad_fea.reshape(mix_grad_fea.shape[0], -1)
        corr = torch.corrcoef(fea_mix)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.min(torch.real(values))
        result_list.append(result)
        meco_layers.append(result.item())
        eigen_layers.append(torch.real(values).tolist())
  

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)
    x=torch.randn(size=x.shape).to('cuda')
    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        
        in_  = x[st:en]
        # print(in_)
        in_ = torch.tensor(in_, requires_grad=True)
        y = net(in_)
        loss = lossfunc(y, target[st:en])
        loss.backward()
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item(),meco_layers,eigen_layers




def get_score_gradoffeature(net, x, target, device, split_data):
    result_list = []
    for param in net.parameters():
        param.requires_grad = True
    def forward_hook(module, data_input, data_output):
        dydx = torch.autograd.grad(outputs=data_output, inputs=data_input,
            grad_outputs=torch.ones_like(data_output),
            retain_graph=True)[0]

        # fea = data_output[0].detach()
        grad = dydx[0].detach()
        # print(grad)
        # if fea.shape == grad.shape:
        #     # print(fea.shape)
        #     # print(grad.shape)
        #     # print('vo day nhe')
        #     # print('orginal:',fea)
        #     mix_grad_fea = torch.cat([fea, grad], dim=1)
        #     # mix_grad_fea = fea*grad
        #     # print('mix', grad)
        # else:
        #     mix_grad_fea = fea
        fea_grad = grad.reshape(grad.shape[0], -1)
        corr = torch.corrcoef(fea_grad)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.min(torch.real(values))
        result_list.append(result)
  

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)

    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        
        in_  = x[st:en]
        # print(in_)
        in_ = torch.tensor(in_, requires_grad=True)
        y = net(in_)
        loss = lossfunc(y, target[st:en])
        loss.backward()
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item()

def get_score_gradoffeature_result(net, x, target, device, split_data):
    result_list = []
    meco_layers = []
    eigen_layers = []
    for param in net.parameters():
        param.requires_grad = True
    def forward_hook(module, data_input, data_output):
        dydx = torch.autograd.grad(outputs=data_output, inputs=data_input,
            grad_outputs=torch.ones_like(data_output),
            retain_graph=True)[0]

        # fea = data_output[0].detach()
        grad = dydx[0].detach()
        # print(grad)
        # if fea.shape == grad.shape:
        #     # print(fea.shape)
        #     # print(grad.shape)
        #     # print('vo day nhe')
        #     # print('orginal:',fea)
        #     mix_grad_fea = torch.cat([fea, grad], dim=1)
        #     # mix_grad_fea = fea*grad
        #     # print('mix', grad)
        # else:
        #     mix_grad_fea = fea
        fea_grad = grad.reshape(grad.shape[0], -1)
        corr = torch.corrcoef(fea_grad)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.min(torch.real(values))
        result_list.append(result)
        meco_layers.append(result.item())
        eigen_layers.append(torch.real(values).tolist())
  

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)

    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        
        in_  = x[st:en]
        # print(in_)
        in_ = torch.tensor(in_, requires_grad=True)
        y = net(in_)
        loss = lossfunc(y, target[st:en])
        loss.backward()
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item(),meco_layers,eigen_layers



def get_score_gradoffeature_input_random_result(net, x, target, device, split_data):
    result_list = []
    meco_layers = []
    eigen_layers = []
    for param in net.parameters():
        param.requires_grad = True
    def forward_hook(module, data_input, data_output):
        dydx = torch.autograd.grad(outputs=data_output, inputs=data_input,
            grad_outputs=torch.ones_like(data_output),
            retain_graph=True)[0]

        # fea = data_output[0].detach()
        grad = dydx[0].detach()
        # print(grad)
        # if fea.shape == grad.shape:
        #     # print(fea.shape)
        #     # print(grad.shape)
        #     # print('vo day nhe')
        #     # print('orginal:',fea)
        #     mix_grad_fea = torch.cat([fea, grad], dim=1)
        #     # mix_grad_fea = fea*grad
        #     # print('mix', grad)
        # else:
        #     mix_grad_fea = fea
        fea_grad = grad.reshape(grad.shape[0], -1)
        corr = torch.corrcoef(fea_grad)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.min(torch.real(values))
        result_list.append(result)
        meco_layers.append(result.item())
        eigen_layers.append(torch.real(values).tolist())
  

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)
    x = torch.randn(size=x.shape)
    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        
        in_  = x[st:en]
        # print(in_)
        in_ = torch.tensor(in_, requires_grad=True)
        y = net(in_)
        loss = lossfunc(y, target[st:en])
        loss.backward()
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item(),meco_layers,eigen_layers

# @measure('meco', bn=True)
# def compute_meco(net, inputs, targets, split_data=1, loss_fn=None):
#     device = inputs.device
#     # Compute gradients (but don't apply them)
#     net.zero_grad()

#     try:
#         meco = get_score(net, inputs, targets, device, split_data=split_data)
#     except Exception as e:
#         print(e)
#         meco = np.nan, None
#     return meco
