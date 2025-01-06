# =========================================================== #
# date: 2024.02.23
# update: 2024.04.26
# authors: updated by sh
# =========================================================== #
import sys
import numpy as np
import json
import argparse
import os
import re
import multiprocessing

# 得到k-mer编码词典
def get_dict(alphabet, k):
    combinations = []

    def generate_helper(current_combination, remaining_length):
        if remaining_length == 0:
            combinations.append(current_combination)
            return
        for letter in alphabet:
            generate_helper(current_combination + letter, remaining_length - 1)

    generate_helper("", k)
    char2id_dict = {c: i for (i, c) in enumerate(combinations)}
    id2char_dict = {i: c for (i, c) in enumerate(combinations)}
    return char2id_dict, id2char_dict




def get_argument_parser():
    parser = argparse.ArgumentParser();
    parser.add_argument('--file_name', type=str, default='HePy',
                        help='The name of the input file')
    parser.add_argument('--dictionary_encoding_k', type=int, default='1',
                        help='The dictionary encoding hyper parameters k')
    parser.add_argument('--dictionary_encoding_w', type=int, default='1',
                        help='The slide window hyper parameters w')
    parser.add_argument('--dictionary_encoding_p', type=int, default='1',
                        help='The parallel hyper parameters p')
    return parser

parser = get_argument_parser()
FLAGS = parser.parse_args()

input_file = FLAGS.file_name
dictionary_encoding_k = FLAGS.dictionary_encoding_k
if dictionary_encoding_k not in [1, 2, 3, 4]:
    print("dictionary_encoding_k Error!")
    exit(0)
dictionary_encoding_w = FLAGS.dictionary_encoding_w
dictionary_encoding_p = FLAGS.dictionary_encoding_p
base_name = os.path.basename(input_file)
# a very import file used to save compression parameters
param_file = "params_" + os.path.splitext(base_name)[0] + "_" + str(dictionary_encoding_k) + "_" + str(dictionary_encoding_w)
output_file = os.path.splitext(base_name)[0] + "_"   + str(dictionary_encoding_k) + "_" + str(dictionary_encoding_w)
print("input_file : ", input_file)
print("(p,w,k)-Mer : ", dictionary_encoding_p, dictionary_encoding_w, dictionary_encoding_k)
print("param_file : ", param_file)
print("output_file : ", output_file)

with open(input_file) as fp:
    data = fp.read()
data = re.sub(r'[^ACGT]', '', data)
print("Seq Length {}".format(len(data)))

char2id_dict, id2char_dict = get_dict(['A','C','G','T'], dictionary_encoding_k)


params = {'char2id_dict':char2id_dict, 'id2char_dict':id2char_dict, 'Write-Chars':""}
def encode_sequence(data, char2id_dict, k, w):
    res = []
    i = 0
    while i < len(data) - k + w:
        sub_ = str(data[i:i + k])
        #print(sub_)
        if len(sub_) == k :
            res.append(char2id_dict[sub_])
        else:
            params['Write-Chars'] = sub_[k-w:]
            #print("Write-Chars: ", sub_[k-w:])

            #return res
        i = i + w
    return res

data = encode_sequence(data, char2id_dict, dictionary_encoding_k, dictionary_encoding_w)
# pre-processing data stream using dictionary encoding using parameter dictionary_encoding_k
#data = [data[i:i+dictionary_encoding_k] for i in range(0, len(data), dictionary_encoding_k)]
#vals = list(set(data))
#vals.sort()


#char2id_dict = {c: i for (i,c) in enumerate(vals)}
#id2char_dict = {i: c for (i,c) in enumerate(vals)}


with open(param_file, 'w') as f:
    json.dump(params, f, indent=4)

#print(char2id_dict)
#print(id2char_dict)

#out = [char2id_dict[c] for c in data]
integer_encoded = np.array(data)
np.save(output_file, integer_encoded)

