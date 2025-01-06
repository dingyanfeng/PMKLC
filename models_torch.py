import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
import math
import numerator_and_denominator as num_and_den

def valid_feature_type(feature_type):
    bool1 = feature_type in ['relu', 'elu+1', 'sqr', 'favor+']
    bool2 = feature_type.startswith('favor+') and feature_type.split(
        '_')[1].isdigit()
    return bool1 or bool2

class SLiMPerformer(torch.nn.Module):

    def __init__(self, vocab_size, vocab_dim, hidden_dim, n_layers, ffn_dim, n_heads, feature_type, compute_type):
        super(SLiMPerformer, self).__init__()

        self._vocab_size = vocab_size
        self._vocab_dim = vocab_dim
        self._hidden_dim = hidden_dim
        self._scale = hidden_dim // vocab_dim       # 等于是一个hidden_dimm能放scale个vocab_dim
        self.input_map = torch.nn.Embedding(vocab_size, vocab_dim // 2)     # 256, 64/2
        self.output_logit_map = torch.nn.Linear(hidden_dim, vocab_size)

        self.layers = torch.nn.ModuleList([
            SLiMPerformerLayer(hidden_dim, ffn_dim, n_heads, feature_type,
                               compute_type) for _ in range(n_layers)
        ])

    def forward(self, x):
        x = self.input_map(x)                                             # 执行词嵌入
        x = self._concat_pos_embs(x, 0)                                   # 位置编码
        bs, seqlen, vlen = x.shape
        x = x.reshape(bs, seqlen // self._scale, vlen * self._scale)      # 在这里使用transformer
        for layer in self.layers:
            x = layer.full_forward(x, layer.attention.sample_rfs(x.device))
        x = self.output_logit_map(x)
        return x

    def full_loss(self, inputs, with_grad=True):
        logits = self.forward(inputs[:, :-1])       # 输入训练数据集
        logits = logits.transpose(1, 2)             # 转置    128 * 256 * 8
        loss = torch.nn.functional.cross_entropy(logits[:, :, -1], inputs[:, -1], reduction='mean')
        if with_grad:
            loss.backward()
        return loss, logits

    def _concat_pos_embs(self, x, start_index):
        pos_emb_size = self._vocab_dim // 2

        positions = torch.arange(
            start_index, start_index + x.shape[1], dtype=x.dtype, device=x.device)
        freqs = torch.exp(
            torch.arange(0, pos_emb_size, 2, dtype=x.dtype, device=x.device) *
            (-np.log(10000) / pos_emb_size))
        args = positions[None, :, None] * freqs[None, None, :]
        sin_pos_embs = torch.sin(args) * torch.ones_like(x[:, :1, :1])
        cos_pos_embs = torch.cos(args) * torch.ones_like(x[:, :1, :1])
        return torch.cat([x, sin_pos_embs, cos_pos_embs], 2)

class SLiMPerformerLayer(torch.nn.Module):

    def __init__(self, hidden_dim, ffn_dim, n_heads, feature_type, compute_type):
        super(SLiMPerformerLayer, self).__init__()
        self.attention = MultiHeadAttention(feature_type, n_heads, hidden_dim,
                                            compute_type)
        self.U_map = torch.nn.Linear(hidden_dim, ffn_dim)
        self.V_map = torch.nn.Linear(ffn_dim, hidden_dim)
        self.layernorm1 = torch.nn.LayerNorm(hidden_dim)
        self.layernorm2 = torch.nn.LayerNorm(hidden_dim)

    def full_forward(self, x, rfs):
        skip = x

        x = self.layernorm1(x)

        x = self.attention.full_forward(x, rfs)

        x = skip + x

        x = self._ffn(x)
        x = self._ffn(x)
        return x

    def _ffn(self, x):      # shared ffn??
        skip = x
        x = self.layernorm2(x)
        x = self.U_map(x)
        x = torch.nn.functional.gelu(x)
        x = self.V_map(x)
        x = skip + x
        return x


class MultiHeadAttention(torch.nn.Module):
    """Explicit multihead attention using prefix sum."""

    def __init__(self, feature_type, n_heads, hidden_dim, compute_type):

        super(MultiHeadAttention, self).__init__()

        self._feature_type = feature_type
        self._n_heads = n_heads
        self._hidden_dim = hidden_dim
        self._compute_type = compute_type

        self.q_map = torch.nn.Linear(hidden_dim, hidden_dim)
        self.k_map = torch.nn.Linear(hidden_dim, hidden_dim)
        self.v_map = torch.nn.Linear(hidden_dim, hidden_dim)

    def full_forward(self, x, rfs):

        queries, keys, values = self._get_queries_keys_values(x, rfs)

        num_sums, den_sums = self.init_sums(x.device)

        if self._compute_type == 'iter':
            num, _ = num_and_den.num_iter(queries, keys, values, num_sums)
            den, _ = num_and_den.den_iter(queries, keys, den_sums)
        elif self._compute_type == 'ps':
            num, _ = num_and_den.num_ps(queries, keys, values, num_sums, False)
            den, _ = num_and_den.den_ps(queries, keys, den_sums, False)
        else:
            num, _ = num_and_den.num_ps(queries, keys, values, num_sums, True)
            den, _ = num_and_den.den_ps(queries, keys, den_sums, True)

        num = torch.transpose(num, 0, 1)
        den = torch.transpose(den, 0, 1)

        outputs = num / (den[Ellipsis, None] + 1e-16)
        outputs = outputs.reshape(x.shape)

        return outputs

    def init_sums(self, device):

        head_dim = self._hidden_dim // self._n_heads

        if self._feature_type.startswith('favor+_'):
            splitted = self._feature_type.split('_')
            feature_dim = int(splitted[1]) * head_dim
        else:
            feature_dim = head_dim

        num_sums = torch.zeros([1, self._n_heads, feature_dim, head_dim], device=device)
        den_sums = torch.zeros([1, self._n_heads, feature_dim], device=device)

        return num_sums, den_sums

    def _get_queries_keys_values(self, inputs, rfs):

        queries = self.q_map(inputs)
        keys = self.k_map(inputs)
        values = self.v_map(inputs)

        queries = queries.reshape(
            [queries.shape[0], queries.shape[1], self._n_heads, -1])
        keys = keys.reshape([keys.shape[0], keys.shape[1], self._n_heads, -1])
        values = values.reshape(
            [values.shape[0], values.shape[1], self._n_heads, -1])

        if self._feature_type == 'relu':
            queries = torch.nn.functional.relu(queries)
            keys = torch.nn.functional.relu(keys)
        elif self._feature_type == 'elu+1':
            queries = torch.nn.functional.elu(queries) + 1
            keys = torch.nn.functional.elu(keys) + 1
        elif self._feature_type == 'sqr':
            queries **= 2
            keys = keys ** 2
        elif self._feature_type == 'abs':
            queries = torch.abs(queries)
            keys = torch.abs(keys)
        else:

            head_dim = self._hidden_dim // self._n_heads

            queries = queries * np.power(head_dim, -0.25)
            queries = torch.einsum('ijkl,klm->ijkm', queries, rfs) - (queries ** 2).sum(
                3, keepdim=True) / 2
            queries = torch.exp(queries)

            keys = keys * np.power(head_dim, -0.25)
            keys = torch.einsum('ijkl,klm->ijkm', keys, rfs) - (keys ** 2).sum(
                3, keepdim=True) / 2
            keys = torch.exp(keys)

        queries = queries.transpose(0, 1)
        keys = keys.transpose(0, 1)
        values = values.transpose(0, 1)

        return queries, keys, values

    def sample_rfs(self, device):

        if not self._feature_type.startswith('favor+'):
            return None

        if self._feature_type == 'favor+':
            factor = 1
        else:
            splitted = self._feature_type.split('_')
            factor = int(splitted[1])

        head_dim = self._hidden_dim // self._n_heads

        rfs = [[
            _sample_orth_matrix(head_dim, device)[None, Ellipsis] for _ in range(factor)
        ] for _ in range(self._n_heads)]
        rfs = [torch.cat(x, 2) for x in rfs]
        rfs = torch.cat(rfs, 0)
        rfs = rfs * np.sqrt(head_dim)

        return rfs

def _sample_orth_matrix(size, device):
    """Samples orthogonal matrix to reduce variance for random features."""
    subspace = torch.randn(size, size, device=device)
    subspace = torch.tril(subspace)
    subspace /= torch.sqrt((subspace ** 2).sum(0, keepdim=True))

    S = torch.triu(subspace.T.mm(subspace)) - 0.5 * torch.eye(
        subspace.shape[1], device=device)

    result = torch.eye(
        subspace.shape[0], device=device) - subspace.mm(torch.inverse(S)).mm(
        subspace.T)

    return result

# mini 模式
class BootstrapNN(nn.Module):
    def __init__(self, vocab_size, emb_size, length, jump, hdim1, hdim2, n_layers, bidirectional):
        super(BootstrapNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.vocab_size = vocab_size
        self.len = length
        self.hdim1 = hdim1  # hidden_dim 1
        self.hdim2 = hdim2
        self.n_layers = n_layers
        self.bidirectional = bidirectional  # True False
        self.jump = jump
        # emb_size = 输入维度
        self.rnn_cell = nn.GRU(emb_size, hdim1, n_layers, batch_first=True, bidirectional=bidirectional)  # 双向

        # length = timesteps
        if bidirectional:
            self.lin1 = nn.Sequential(
                nn.Linear(2 * hdim1 * (length // jump), hdim2),         # 隐藏层分为forward layer和backward layer，两个时间方向的信息传递，隐藏节点翻倍
                nn.ReLU(inplace=True)           # inplace=True 不会新建一个参数，而是替代掉原来的
            )
            self.flin1 = nn.Linear(2 * hdim1 * (length // jump), vocab_size)
        else:
            self.lin1 = nn.Sequential(
                nn.Linear(hdim1 * (length // jump), hdim2),
                nn.ReLU(inplace=True)
            )
            self.flin1 = nn.Linear(hdim1 * (length // jump), vocab_size)
        self.flin2 = nn.Linear(hdim2, vocab_size)

    def forward(self, inp):
        emb = self.embedding(inp)            # [batch-size, length, emb_size] ==> [2048, 64, 8]
        output, hidden = self.rnn_cell(emb)  # [batch-size, length, hidim1*2] ==> [2048, 64, 64]
                                             # [n_layers * 2(bi-GRU), batch-size, hidim1] ==> [4, 2048, 32]
        slicedoutput = torch.flip(output, [1])[:, ::self.jump, :]  # [2048, 4, 64]
        batch_size = slicedoutput.size()[0]
        flat = slicedoutput.contiguous().view(batch_size, -1)      # [batch-size, length/self.jump * hiddim1 * num_directions] ==> [2048,256]
        prelogits = x = self.lin1(flat)       # [batch-size, hidim2] ==> [2018, 16]
        x = self.flin1(flat) + self.flin2(x)  # [batch-size, vocab-size] ===> [2048, 4]
        out = F.log_softmax(x, dim=1)

        return out


# distillation模式
class DistiallationNN(nn.Module):
    def __init__(self, bsNN, kdNN, vocab_dim, hidden_dim, n_layers, ffn_dim, n_heads, feature_type, compute_type, num_iters, model_list):
        super(DistiallationNN, self).__init__()
        self.flag = 0                         # 计数器
        self._model_list = model_list         # [PTM, HTM, SM] 表示具体使用的模型
        if model_list[2] == 1:
            self._hidden_dim = hidden_dim         # 隐藏层的维度
            self._vocab_dim = vocab_dim           # 词嵌入维度
            self._scale = hidden_dim // vocab_dim
            self._vocab_size = bsNN.vocab_size    # 词汇表大小
            self.input_map = torch.nn.Embedding(self._vocab_size, self._vocab_dim // 2)
            self.output_logit_map = torch.nn.Linear(self._hidden_dim, self._vocab_size)
            self.layers = torch.nn.ModuleList([
                SLiMPerformerLayer(self._hidden_dim, ffn_dim, n_heads, feature_type, compute_type) for _ in range(n_layers)])

        if model_list[0] == 1:
            # Public Teacher Model
            self.bsembedding = bsNN.embedding
            self.bsrnn_cell = bsNN.rnn_cell
            self.bslin1 = bsNN.lin1
            self.bsjump = bsNN.jump
            if bsNN.bidirectional:
                self.bsflin1 = bsNN.flin1
            else:
                self.bsflin1 = bsNN.flin1
            self.bsflin2 = bsNN.flin2

        if model_list[1] == 1:
            # Home Teacher Model
            self.kdembedding = kdNN.embedding
            self.kdrnn_cell = kdNN.rnn_cell
            self.kdlin1 = kdNN.lin1
            self.kdjump = kdNN.jump
            if kdNN.bidirectional:
                self.kdflin1 = kdNN.flin1
            else:
                self.kdflin1 = kdNN.flin1
            self.kdflin2 = kdNN.flin2
        if model_list.count(1) == 3:
            self.weights = nn.Parameter(torch.ones(3, dtype=torch.float), requires_grad=True)
        elif model_list.count(1) == 2:
            self.weights = nn.Parameter(torch.ones(2, dtype=torch.float), requires_grad=True)

    def forward(self, inp):
        self.flag = self.flag + 1

        if self._model_list[1] == 1:
            emb = self.kdembedding(inp)
            output, hidden = self.kdrnn_cell(emb)
            slicedoutput = torch.flip(output, [1])[:, ::self.kdjump, :]
            batch_size = slicedoutput.size()[0]
            flat = slicedoutput.contiguous().view(batch_size, -1)
            x = self.kdlin1(flat)
            kd_logits = self.kdflin1(flat) + self.kdflin2(x)

        if self._model_list[0] == 1:
            emb = self.bsembedding(inp)
            output, hidden = self.bsrnn_cell(emb)
            slicedoutput = torch.flip(output, [1])[:, ::self.bsjump, :]
            batch_size = slicedoutput.size()[0]
            flat = slicedoutput.contiguous().view(batch_size, -1)
            x = self.bslin1(flat)
            old_logits = self.bsflin1(flat) + self.bsflin2(x)
        if self._model_list[2] == 1:
            x = self.input_map(inp)           # 执行词嵌入
            x = self._concat_pos_embs(x, 0)   # 执行位置嵌入
            bs, seqlen, vlen = x.shape
            x = x.reshape(bs, seqlen // self._scale, vlen * self._scale)
            for layer in self.layers:
                x = layer.full_forward(x, layer.attention.sample_rfs(x.device))
            x = self.output_logit_map(x)
            new_logits = x.transpose(1, 2)[:, :, -1]
        if self._model_list.count(1) == 3:
            normalized_weights = F.softmax(self.weights, dim=0)
            final_logits = normalized_weights[0]*new_logits + normalized_weights[1]*old_logits + normalized_weights[2]*kd_logits
        elif self._model_list[0] == 1 and self._model_list[1] == 1:
            normalized_weights = F.softmax(self.weights, dim=0)
            final_logits = normalized_weights[0]*old_logits + normalized_weights[1]*kd_logits
        elif self._model_list[0] == 1 and self._model_list[2] == 1:
            normalized_weights = F.softmax(self.weights, dim=0)
            final_logits = normalized_weights[0]*old_logits + normalized_weights[1]*new_logits
        elif self._model_list[1] == 1 and self._model_list[2] == 1:
            normalized_weights = F.softmax(self.weights, dim=0)
            final_logits = normalized_weights[0]*kd_logits + normalized_weights[1]*new_logits
        elif self._model_list[0] == 1:
            final_logits = old_logits
        elif self._model_list[1] == 1:
            final_logits = kd_logits
        else:
            final_logits = new_logits
        out = F.log_softmax(final_logits, dim=1)
        if self._model_list[2] == 1:
            return out, F.log_softmax(new_logits, dim=1)
        else:
            return out, -1
    
    def _concat_pos_embs(self, x, start_index):
        pos_emb_size = self._vocab_dim // 2

        positions = torch.arange(
            start_index, start_index + x.shape[1], dtype=x.dtype, device=x.device)
        freqs = torch.exp(
            torch.arange(0, pos_emb_size, 2, dtype=x.dtype, device=x.device) *
            (-np.log(10000) / pos_emb_size))
        args = positions[None, :, None] * freqs[None, None, :]
        sin_pos_embs = torch.sin(args) * torch.ones_like(x[:, :1, :1])
        cos_pos_embs = torch.cos(args) * torch.ones_like(x[:, :1, :1])
        return torch.cat([x, sin_pos_embs, cos_pos_embs], 2)
