import numpy as np
import os
import sys
import json
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from models_torch import *
from utils import *
import tempfile
import argparse
import arithmeticcoding_fast
import struct
import time
import shutil

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(3407)

def loss_function(pred, target):
    loss = 1/np.log(2) * F.nll_loss(pred, target)
    return loss


def decompress(model, len_series, bs, vocab_size, timesteps, device, optimizer, scheduler, final_step=False):
    
    if not final_step:
        num_iters = len_series // bs
        series_2d = np.zeros((bs,num_iters), dtype = np.uint8).astype('int')
        ind = np.array(range(bs))*num_iters

        f = [open(FLAGS.temp_file_prefix+'.'+str(i),'rb') for i in range(bs)]
        bitin = [arithmeticcoding_fast.BitInputStream(f[i]) for i in range(bs)]
        dec = [arithmeticcoding_fast.ArithmeticDecoder(32, bitin[i]) for i in range(bs)]

        prob = np.ones(vocab_size)/vocab_size
        cumul = np.zeros(vocab_size+1, dtype = np.uint64)
        cumul[1:] = np.cumsum(prob*10000000 + 1)

        # Decode first K symbols in each stream with uniform probabilities
        for i in range(bs):
            for j in range(min(timesteps, num_iters)):
                series_2d[i,j] = dec[i].read(cumul, vocab_size)

        cumul = np.zeros((bs, vocab_size+1), dtype = np.uint64)

        block_len = 1
        test_loss = 0
        batch_loss = 0
        start_time = time.time()

        if FLAGS.load:
            print(f"GPU-{FLAGS.gpu} Load model from {int(FLAGS.gpu)-1}")
            model.load_state_dict(torch.load(FLAGS.file_name + '_model/{}.{}.pth'.format(FLAGS.file_name, int(FLAGS.gpu)-1)))
        for j in (range(num_iters - timesteps)):
            # Create Batch
            bx = Variable(torch.from_numpy(series_2d[:,j:j+timesteps])).to(device)
            
            with torch.no_grad():
                model.eval()
                pred, _ = model(bx)
                prob = torch.exp(pred).detach().cpu().numpy()
            cumul[:,1:] = np.cumsum(prob*10000000 + 1, axis = 1)

            # Decode with Arithmetic Encoder
            for i in range(bs):
                series_2d[i,j+timesteps] = dec[i].read(cumul[i,:], vocab_size)
            
            by = Variable(torch.from_numpy(series_2d[:, j+timesteps])).to(device)
            loss = loss_function(pred, by)
            test_loss += loss.item()
            batch_loss += loss.item()

            # Update Parameters of Combined Model
            if (j+1) % block_len == 0 and FLAGS.model_list[2] == 1:
                model.train()
                optimizer.zero_grad()
                data_x = np.concatenate([series_2d[:, j + np.arange(timesteps) - p] for p in range(block_len)], axis=0)
                data_y = np.concatenate([series_2d[:, j + timesteps - p] for p in range(block_len)], axis=0)

                bx = Variable(torch.from_numpy(data_x)).to(device)
                by = Variable(torch.from_numpy(data_y)).to(device)
                pred1, pred2 = model(bx)
                loss2 = loss_function(pred2, by)
                loss = loss_function(pred1, by) + loss2
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                # with torch.no_grad():
                #     model.T.clamp_(0, 2)

            if j == int(FLAGS.ratio * (num_iters - timesteps)) and FLAGS.save:
                print(f"GPU-{FLAGS.gpu} Save model with pre-train ratio {FLAGS.ratio}")
                torch.save(model.state_dict(), f'{FLAGS.file_name}_model/{FLAGS.file_name}.{FLAGS.gpu}.pth')

        if num_iters <= timesteps and FLAGS.save:
            print(f"GPU-{FLAGS.gpu} Save model default")
            torch.save(model.state_dict(), f'{FLAGS.file_name}_model/{FLAGS.file_name}.{FLAGS.gpu}.pth')


        # close files
        for i in range(bs):
            bitin[i].close()
            f[i].close()
        return series_2d.reshape(-1)
    
    else:
        series = np.zeros(len_series, dtype = np.uint8).astype('int')
        f = open(FLAGS.temp_file_prefix+'.last','rb')
        bitin = arithmeticcoding_fast.BitInputStream(f)
        dec = arithmeticcoding_fast.ArithmeticDecoder(32, bitin)
        prob = np.ones(vocab_size)/vocab_size
        cumul = np.zeros(vocab_size+1, dtype = np.uint64)
        cumul[1:] = np.cumsum(prob*10000000 + 1)        

        for j in range(min(timesteps,len_series)):
            series[j] = dec.read(cumul, vocab_size)
        for i in range(len_series-timesteps):
            bx = Variable(torch.from_numpy(series[i:i+timesteps].reshape(1,-1))).to(device)
            with torch.no_grad():
                model.eval()
                pred, _ = model(bx)
                prob = torch.exp(pred).detach().cpu().numpy()
            cumul[1:] = np.cumsum(prob*10000000 + 1)
            series[i+timesteps] = dec.read(cumul, vocab_size)
        bitin.close()
        f.close()
        return series

def get_argument_parser():
    parser = argparse.ArgumentParser();
    parser.add_argument('--k', type=int, default='3',
                        help='The value of k')
    parser.add_argument('--w', type=int, default='3',
                        help='The value of w')
    parser.add_argument('--file_name', type=str, default='ScPo_3_3.pmklc',
                        help='The name of the input file')
    parser.add_argument('--output', type=str, default='ScPo_3_3.pmklc_recover',
                        help='The name of the output file')
    parser.add_argument('--model_list_num', type=str, default='7',
                        help='Which models to use, 1-7')
    parser.add_argument('--static_public_model_path', type=str, default='SPuM/all_3_3.out',
                        help='Path to static public model weights')
    parser.add_argument('--static_private_model_path', type=str, default='SPrM/ScPo_SPrM_3_3',
                        help='Path to static private model parameters')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU to use')
    parser.add_argument('--model_catena', type=str, default='SPuM',
                        help='the used model catena')
    parser.add_argument('--save', action='store_true', help='Save the model')
    parser.add_argument('--load', action='store_true', help='Load the model')
    parser.add_argument('--ratio', type=float, default=0.05, help='Pretrain ratio.')
    return parser


def var_int_decode(f):
    byte_str_len = 0
    shift = 1
    while True:
        this_byte = struct.unpack('B', f.read(1))[0]
        byte_str_len += (this_byte & 127) * shift
        if this_byte & 128 == 0:
                break
        shift <<= 7
        byte_str_len += shift
    return byte_str_len

def main():
    os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu
    use_cuda = True

    FLAGS.temp_dir = str(FLAGS.gpu) + 'temp'
    if os.path.exists(FLAGS.temp_dir):
        shutil.rmtree(str(FLAGS.gpu) + 'temp')
    FLAGS.temp_file_prefix = FLAGS.temp_dir + "/compressed"
    if not os.path.exists(FLAGS.temp_dir):
        os.makedirs(FLAGS.temp_dir)

    f = open(FLAGS.file_name+'.params','r')
    params = json.loads(f.read())
    f.close()

    batch_size = params['bs']
    timesteps = params['timesteps']
    len_series = params[f'len_series{FLAGS.gpu}']
    id2char_dict = params['id2char_dict']
    vocab_size = len(id2char_dict)

    # Break into multiple streams
    f = open(FLAGS.gpu+'_'+FLAGS.file_name+'.combined','rb')
    for i in range(batch_size):
        f_out = open(FLAGS.temp_file_prefix+'.'+str(i),'wb')
        byte_str_len = var_int_decode(f)
        byte_str = f.read(byte_str_len)
        f_out.write(byte_str)
        f_out.close()
    f_out = open(FLAGS.temp_file_prefix+'.last','wb')
    byte_str_len = var_int_decode(f)
    byte_str = f.read(byte_str_len)
    f_out.write(byte_str)
    f_out.close()
    f.close()

    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    series = np.zeros(len_series,dtype=np.uint8)

    model_list_num = int(FLAGS.model_list_num)
    FLAGS.model_list = int_to_binary_list(model_list_num)

    bsdic = {'vocab_size': vocab_size, 'emb_size': 16,
        'length': timesteps, 'jump': 16,
        'hdim1': 128, 'hdim2': 128, 'n_layers': 2,
        'bidirectional': True
             }
    
    kddic = {'vocab_size': vocab_size, 'emb_size': 16,
        'length': timesteps, 'jump': 16,
        'hdim1': 128, 'hdim2': 128, 'n_layers': 1,
        'bidirectional': True
             }

    num_iters_ = (params[f'len_series{FLAGS.gpu}'] + timesteps) // batch_size

    finaldic = {'num_iters': num_iters_, 'vocab_dim': 64, 'hidden_dim': 256, 'n_layers': 1,
                'ffn_dim': 4096, 'n_heads': 8, 'feature_type': 'sqr', 'compute_type': 'iter', 'model_list': FLAGS.model_list}


    # Define Model and load bootstrap weights
    bsmodel = BootstrapNN(**bsdic).to(device)
    # print(FLAGS.static_public_model_path)
    if model_list_num >= 4:
        bsmodel.load_state_dict(torch.load(FLAGS.static_public_model_path))
    finaldic['bsNN'] = bsmodel

    kdmodel = BootstrapNN(**kddic).to(device)
    # print("FLAGS.static_private_model_path : ", FLAGS.static_private_model_path)
    if model_list_num!=1 and model_list_num!=4 and model_list_num!=5:
        kdmodel.load_state_dict(torch.load(FLAGS.static_private_model_path))
    finaldic['kdNN'] = kdmodel

    commodel = DistiallationNN(**finaldic).to(device)
    
    # Freeze Bootstrap Weights
    for name, p in commodel.named_parameters():
        if "bs" in name:
            p.requires_grad = False
        elif "kd" in name:
            p.requires_grad = False
    
    # Optimizer
    optimizer = optim.Adam(commodel.parameters(), lr=5e-4, betas=(0.0, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, threshold=1e-2, patience=1000, cooldown=10000, min_lr=1e-4, verbose=True)
    l = int(len(series)/batch_size)*batch_size
    
    series[:l] = decompress(commodel, l, batch_size, vocab_size, timesteps, device, optimizer, scheduler)
    if l < len_series - timesteps:
        series[l:] = decompress(commodel, len_series-l, 1, vocab_size, timesteps, device, optimizer, scheduler, final_step = True)
    else:
        f = open(FLAGS.temp_file_prefix+'.last','rb')
        bitin = arithmeticcoding_fast.BitInputStream(f)
        dec = arithmeticcoding_fast.ArithmeticDecoder(32, bitin) 
        prob = np.ones(vocab_size)/vocab_size
        
        cumul = np.zeros(vocab_size+1, dtype = np.uint64)
        cumul[1:] = np.cumsum(prob*10000000 + 1)        
        for j in range(l, len_series):
            series[j] = dec.read(cumul, vocab_size)
        
        bitin.close() 
        f.close()

    # Write to output
    params['Write-Chars']=""
    merged_string = recover_data(FLAGS.k, FLAGS.w, id2char_dict, series, params['Write-Chars'])
    with open(FLAGS.output, "w") as file_w:
        file_w.write(merged_string)

    shutil.rmtree(str(FLAGS.gpu) + 'temp')
    print(f'GPU-{FLAGS.gpu} compression task done, Peak GPU memory usage: {torch.cuda.max_memory_allocated() // 1024} KBs')
    

if __name__ == "__main__":
    parser = get_argument_parser()
    FLAGS = parser.parse_args()
    main()




