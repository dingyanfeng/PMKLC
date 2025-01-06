import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

class CustomDL(Dataset):

  def __init__(self, features, labels):

    self.features = features
    self.labels = labels

  def __len__(self):
    return len(self.features)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
        idx = idx.tolist()
    feat = self.features[idx]
    lab = self.labels[idx]
    sample = {'x': feat, 'y': lab}

    return sample

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n), writeable=False)

def generate_single_output_data(series,batch_size,time_steps):
    series = series.reshape(-1)
    series = series.copy()
    data = strided_app(series, time_steps+1, 1)
    l = int(len(data)/batch_size) * batch_size

    data = data[:l] 
    X = data[:, :-1]
    Y = data[:, -1]

    return X,Y

def recover_data(k, w, id2char_dict, series, additional_str):
    # 用于解压缩恢复K-Mer编码
    if (k == w):
        res = [id2char_dict[str(s)] for s in series]
        merged_string = ''.join(res)
        if len(additional_str) > 0:  # k - w 是交叉重叠的部分
            merged_string = merged_string + additional_str
        return merged_string
    else:

        res = [0] * len(series)
        #res = [id2char_dict[str(s)] for s in series]
        i = 0
        while (i != len(res)):
            if (i == 0):
                res[i] = id2char_dict[str(series[i])]
            else:
                res[i] = id2char_dict[str(series[i])][k-w:]
            i = i + 1
        merged_string = ''.join(res)
        if len(additional_str) > 0:  # k - w 是交叉重叠的部分
            merged_string = merged_string + additional_str
        return merged_string


def int_to_binary_list(number):
    if 1 <= number <= 7:
        binary_str = bin(number)[2:]
        
        binary_str = binary_str.zfill(3)
        
        binary_list = [int(digit) for digit in binary_str]
    else:
        raise ValueError("Number must be between 1 and 7")
    
    return binary_list


