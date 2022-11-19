import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import math
from typing import Optional
# from deepspeed.ops.sparse_attention import SparseSelfAttention, SparsityConfig, FixedSparsityConfig


class RNN(nn.Module):
    # RNN model is composed of three parts: a word embedding layer, a rnn network and a output layer
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding
    # The network has input of each word embedding and output a hidden feature corresponding to each word embedding
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary
    # feel free to change the init arguments if necessary
    def __init__(self, nvoc, ninput, nhid, nlayers, rnnType, device, args):
        super(RNN, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.device = device
        self.rnnType = rnnType
        # self.embed change input size BxL into BxLxE
        self.embed = nn.Embedding(nvoc, ninput)
        self.args = args
        self.ninput = ninput

        # WRITE CODE HERE witnin two '#' bar                                              #
        # Construct you RNN model here. You can add additional parameters to the function #
        ###################################################################################
        if rnnType == "GRU":
            self.rnn = nn.GRU(input_size=ninput, hidden_size=nhid, num_layers=nlayers)
        elif rnnType == "LSTM":
            self.rnn = LSTM(num_inputs=ninput, num_hiddens=nhid, device=self.device)
        elif rnnType == "Transformer":
            self.rnn = LMTransformer(nvoc=nvoc, ninput=ninput, nhead=8, d_hid=nhid, nlayers=nlayers, device=self.device,
                                     args=args)

        self.state = None
        ###################################################################################
        self.decoder = nn.Linear(nhid, nvoc)
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers
        self.src_mask = generate_square_subsequent_mask(args.max_sql).to(self.device)

    def init_weights(self):
        init_uniform = 0.1
        self.embed.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    # feel free to change the forward arguments if necessary
    def forward(self, input):

        # WRITE CODE HERE within two '#' bar                                             #
        # With embeddings, you can get your output here.                                 #
        # Output has the dimension of sequence_length * batch_size * number of classes   #
        ##################################################################################
        if self.rnnType == "GRU" or self.rnnType == "LSTM":
            embeddings = self.drop(self.embed(input))
            output, state = self.rnn(embeddings, self.state)
            hidden = state
            output = self.drop(output)
            decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        if self.rnnType == "Transformer":
            embeddings = self.drop(self.embed(input)) * math.sqrt(self.ninput)
            seq_len = input.size(0)

            if seq_len != self.args.max_sql:  # only on last batch
                src_mask = self.src_mask[:seq_len, :seq_len]
                output = self.rnn(embeddings, src_mask)
            else:
                output = self.rnn(embeddings, self.src_mask)

            hidden = None
            decoded = output.view(output.size(0) * output.size(1), output.size(2))

        ##################################################################################

        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden


# WRITE CODE HERE within two '#' bar                                                      #
# your LSTM for language modeling implmentation here                               #
###########################################################################################
class LSTM(nn.Module):
    def __init__(self, num_inputs, num_hiddens, device, nlayers=1):
        super(LSTM, self).__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_outputs = num_hiddens
        self.nlayers = nlayers
        self.device = device
        self.W_xi, self.W_hi, self.b_i, self.W_xf, self.W_hf, self.b_f, self.W_xo, self.W_ho, self.b_o, self.W_xc, self.W_hc, self.b_c, self.W_hq, self.b_q = self.get_params()

    def get_params(self):
        def _one(shape):
            ts = torch.tensor(np.random.normal(0, 0.01, size=shape), dtype=torch.float32, device=self.device)
            return torch.nn.Parameter(ts, requires_grad=True)

        def _three():
            return (_one((self.num_inputs, self.num_hiddens)),
                    _one((self.num_hiddens, self.num_hiddens)),
                    torch.nn.Parameter(torch.zeros(self.num_hiddens, dtype=torch.float32, device=self.device),
                                       requires_grad=True))

        W_xi, W_hi, b_i = _three()  # 输入门参数
        W_xf, W_hf, b_f = _three()  # 遗忘门参数
        W_xo, W_ho, b_o = _three()  # 输出门参数
        W_xc, W_hc, b_c = _three()  # 候选记忆细胞参数

        # 输出层参数
        W_hq = _one((self.num_hiddens, self.num_outputs))
        b_q = torch.nn.Parameter(torch.zeros(self.num_outputs, dtype=torch.float32, device=self.device),
                                 requires_grad=True)

        return nn.ParameterList([W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q])

    def forward(self, inputs, state):
        if state is None:
            state = (torch.zeros((inputs.size()[1], self.num_hiddens), device=self.device),
                     torch.zeros((inputs.size()[1], self.num_hiddens), device=self.device))
        (H, C) = state
        outputs = []
        for X in inputs:
            I = torch.sigmoid(torch.matmul(X, self.W_xi) + torch.matmul(H, self.W_hi) + self.b_i)
            F = torch.sigmoid(torch.matmul(X, self.W_xf) + torch.matmul(H, self.W_hf) + self.b_f)
            O = torch.sigmoid(torch.matmul(X, self.W_xo) + torch.matmul(H, self.W_ho) + self.b_o)
            C_tilda = torch.tanh(torch.matmul(X, self.W_xc) + torch.matmul(H, self.W_hc) + self.b_c)
            C = F * C + I * C_tilda
            H = O * C.tanh()
            Y = torch.matmul(H, self.W_hq) + self.b_q
            outputs.append(Y)

        outputs = torch.stack(outputs)
        outputs.to(self.device)
        return outputs, (H, C)


###########################################################################################


# WRITE CODE HERE within two '#' bar                                                      #
# your transformer for language modeling implmentation here                               #
###########################################################################################
class LMTransformer(nn.Module):
    def __init__(self, nvoc, ninput, nhead, d_hid, device, args,
                 nlayers: int, dropout: float = 0.2):
        super(LMTransformer, self).__init__()
        self.ntoken = nvoc
        self.d_model = ninput
        self.model_type = 'Transformer'
        self.device = device
        self.nlayers = nlayers
        self.pos_encoder = PositionalEncoder(self.d_model, dropout=dropout)

        self.nhead = nhead

        encoder_layers = nn.TransformerEncoderLayer(self.d_model, nhead, d_hid, dropout)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

        self.transformer_encoder = TransformerEncoder(nlayers, input_dim=ninput, num_heads=nhead, dim_feedforward=d_hid,
                                                      dropout=dropout)

        self.decoder = nn.Linear(self.d_model, self.ntoken)


        self.args = args
        maxpos = args.max_sql
        attn_heads = nhead
        self.slopes = torch.Tensor(self.get_slopes(attn_heads))
        # In the next line, the part after the * is what constructs the diagonal matrix (right matrix in Figure 3 in the paper).
        # If you run it you'll see that it doesn't exactly print out the same matrix as we have in Figure 3, but one where all rows are identical.
        # This works because the softmax operation is invariant to translation, and our bias functions are always linear.
        self.alibi = self.slopes.unsqueeze(1).unsqueeze(1) * torch.arange(maxpos).unsqueeze(0).unsqueeze(0).expand(
            attn_heads, -1, -1)
        self.alibi = self.alibi.view(attn_heads, 1, maxpos)
        if self.training:
            self.alibi = self.alibi.repeat(args.train_batch_size, 1, 1).to(self.device) # batch_size, 1, 1
        else:
            self.alibi = self.alibi.repeat(args.eval_batch_size, 1, 1).to(self.device)  # batch_size, 1, 1
        self._future_mask = torch.empty(0).to(self.device)
        self.init_weights()



    def fill_with_neg_inf(self,t):
        """FP16-compatible function that fills a tensor with -inf."""
        return t.float().fill_(float("-inf")).type_as(t)
    def get_slopes(self,n):
        def get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(
                n)  # In the paper, we only train models that have 2^a heads for some a. This function has
        else:  # some good properties that only occur when the input is a power of 2. To maintain that even
            closest_power_of_2 = 2 ** math.floor(
                math.log2(n))  # when the number of heads is not a power of 2, we use this workaround.
            return get_slopes_power_of_2(closest_power_of_2) + self.get_slopes(2 * closest_power_of_2)[0::2][
                                                               :n - closest_power_of_2]

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
                self._future_mask.size(0) == 0
                or (not self._future_mask.device == tensor.device)
                or self._future_mask.size(1) < self.args.max_sql
        ):
            self._future_mask = torch.triu(
                self.fill_with_neg_inf(torch.zeros([self.args.max_sql, self.args.max_sql])), 1
            ).to(self.device)
            self._future_mask = self._future_mask.unsqueeze(0) + self.alibi
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:tensor.shape[1] * self.nhead, :dim, :dim]

    def fill_with_neg_inf(self,t):
        """FP16-compatible function that fills a tensor with -inf."""
        return t.float().fill_(float("-inf")).type_as(t)
    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:


        # src_mask = self.buffered_future_mask(src)

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)



        return output

    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoder(nn.Module):
    def __init__(self, dim, max_seq_len=5000, dropout=0.1):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # matrix of pos encoding for max_len inputs
        # pe: 96 x 48
        pe = torch.zeros(max_seq_len, dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).reshape(max_seq_len, 1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


###########################################################################################
class EncoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = nn.MultiheadAttention(input_dim, num_heads)

        # self.self_attn = SparseSelfAttention(sparsity_config=FixedSparsityConfig(num_heads=num_heads))

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out, attn_map = self.self_attn(x, x, x, attn_mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, x, x, attn_mask=mask)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps
