import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_io import load_runinfo_from_rundir
from data_tools import data_train_test_split_linear
from settings import DIR_MODELS
from torch_device import device_select


class ReturnLastToken(nn.Module):
    """
    Baseline model -- return final token
    """
    def __init__(self):
        super().__init__()

    def forward(self, xs):
        outs = xs[:, :, -1]  # return the last token
        return outs


def weight_matrix(dim_in, dim_out, mode="default"):
    """
    Can use to initialize weight matrices in nn layers 
        e.g. self.W_v = weight_matrix(h=ndim, w=ndim, mode="default")

    Throughout, we multiply on the right (e.g. y = W @ x) for consistency with the math notation.
        Thus, dim_in is the number of columns, and dim_out is the number of rows. (i.e. w, h in PyTorch notation)

    For info on default init from torch method nn.Linear, see here:
      https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    """
    W_tensor = torch.empty(dim_out, dim_in)
    if mode == "default":
        low  = -1.0 / np.sqrt(dim_in)
        high =  1.0 / np.sqrt(dim_in)
        torch.nn.init.uniform_(W_tensor, a=low, b=high)
    elif mode == "kaiming":
        torch.nn.init.kaiming_uniform_(W_tensor)
    elif mode == "normal":
        torch.nn.init.normal_(W_tensor, mean=0, std=0.02)
    else:
        raise ValueError("Unsupported `mode`")
    return torch.nn.Parameter(W_tensor)


class TransformerModelV1(nn.Module):
    """
    Simplest model:
    - no positional encoding is used
    - `linear self-attention` (no softmax wrapper used)

    Notes
     - dim_input - the dimension of input tokens
     - dim_attn  - the dimension of the residual stream (attention head + MLP input and output)
    """
    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__()
        assert n_layer == 1
        assert n_head == 1
        assert dim_attn is None

        # attention matrices (need to split by head...)
        self.W_KQ = weight_matrix(dim_input, dim_input, mode='default')
        self.W_PV = weight_matrix(dim_input, dim_input, mode='default')
        self.rho = context_length                     # scaling used in Bartlett 2023

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ
        W_PV = self.W_PV
        rho = n_tokens

        attn_arg = torch.transpose(xs, 1, 2) @ W_KQ @ xs / rho
        f_attn = xs + W_PV @ xs @ attn_arg

        out = f_attn[:, :, -1]  # take dim_n output result at last token, for all batches
        return out


class TransformerModelV1nores(TransformerModelV1):
    """
    See docstring TransformerModelV1
    """
    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__(context_length, dim_input, dim_attn=dim_attn, n_layer=n_layer, n_head=n_head)

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ
        W_PV = self.W_PV
        rho = n_tokens

        attn_arg = torch.transpose(xs, 1, 2) @ W_KQ @ xs / rho
        f_attn = W_PV @ xs @ attn_arg  # the residual stream term "+ xs" has been removed

        out = f_attn[:, :, -1]  # take dim_n output result at last token, for all batches
        return out


class TransformerModelV1noresForceDiag(nn.Module):
    """
    See docstring TransformerModelV1
    """
    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__()
        assert n_layer == 1
        assert n_head == 1
        assert dim_attn is None

        # attention matrices (need to split by head...)
        #self.W_KQ = weight_matrix(dim_input, dim_input, mode='normal')
        #self.W_PV = weight_matrix(dim_input, dim_input, mode='normal')

        self.W_KQ = torch.nn.Parameter(torch.tensor(0.1))
        self.W_PV = torch.nn.Parameter(torch.tensor(0.1))

        #self.W_KQ = torch.nn.Parameter(0.1 * torch.eye(dim_input))
        #self.W_PV = torch.nn.Parameter(0.1 * torch.eye(dim_input))
        self.rho = context_length                     # scaling used in Bartlett 2023

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ * torch.eye(n_dim)  # self.W_KQ is a 1-parameter scalar --> make n x n diag arr
        W_PV = self.W_PV * torch.eye(n_dim)  # self.W_PV is a 1-parameter scalar --> make n x n diag arr

        rho = n_tokens

        attn_arg = torch.transpose(xs, 1, 2) @ W_KQ @ xs / rho
        f_attn = W_PV @ xs @ attn_arg  # the residual stream term "+ xs" has been removed

        out = f_attn[:, :, -1]  # take dim_n output result at last token, for all batches
        return out


class TransformerModelV1noresOmitLast(TransformerModelV1):
    """
    See docstring TransformerModelV1
    """
    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__(context_length, dim_input, dim_attn=dim_attn, n_layer=n_layer, n_head=n_head)

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ
        W_PV = self.W_PV
        rho = n_tokens - 1

        xs_skip_last = xs[:, :, :-1]

        projection_estimate = xs_skip_last @ torch.transpose(xs_skip_last, 1, 2) / rho

        f_attn_approx = W_PV @ projection_estimate @ W_KQ @ xs[:, :, [-1]]
        out = f_attn_approx[:, :, -1]  # take dim_n output result at last token, for all batches

        return out


class TransformerModelV1noresForceDiagAndOmitLast(nn.Module):
    """
    See docstring TransformerModelV1
    """
    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__()
        assert n_layer == 1
        assert n_head == 1
        assert dim_attn is None

        # attention matrices (need to split by head...)
        self.W_KQ = torch.nn.Parameter(torch.tensor(0.1))
        self.W_PV = torch.nn.Parameter(torch.tensor(0.1))
        self.rho = context_length

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ * torch.eye(n_dim)  # self.W_KQ is a 1-parameter scalar --> make n x n diag arr
        W_PV = self.W_PV * torch.eye(n_dim)  # self.W_PV is a 1-parameter scalar --> make n x n diag arr

        rho = n_tokens - 1

        xs_skip_last = xs[:, :, :-1]

        #attn_arg = torch.transpose(xs_skip_last, 1, 2) @ W_KQ @ xs_skip_last / rho
        projection_estimate = xs_skip_last @ torch.transpose(xs_skip_last, 1, 2) / rho

        f_attn_approx = W_PV @ projection_estimate @ W_KQ @ xs
        out = f_attn_approx[:, :, -1]  # take dim_n output result at last token, for all batches

        return out


class TransformerModelV2(nn.Module):
    """
    Simplest model:
    - no positional encoding is used
    - same as V1 but now softmax in place of `linear` self-attention
    """
    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__()
        assert n_layer == 1      # TODO implement...
        assert n_head == 1       # TODO implement...
        assert dim_attn is None  # TODO implement... for now we take dim_attn == dim_input
        # TODO in multilayer version, add AttnHead class beneath AttnLayer class? forward pass is just loop over nlayer

        # attention matrices (need to split by head...)
        self.W_KQ = weight_matrix(dim_input, dim_input, mode='default')
        self.W_PV = weight_matrix(dim_input, dim_input, mode='default')
        self.rho = 1.0

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ
        W_PV = self.W_PV

        # new line: now scaling is a fixed constant as in original QKV-attention - 1/sqrt(n)
        attn_arg = torch.transpose(xs, 1, 2) @ W_KQ @ xs / self.rho
        softmax_attn_arg = torch.softmax(attn_arg, dim=1)
        f_attn = xs + W_PV @ xs @ softmax_attn_arg

        out = f_attn[:, :, -1]  # take dim_n output result at last token, for all batches
        return out


class TransformerModelV2nores(TransformerModelV2):
    """
    See docstring TransformerModelV2
    """
    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__(context_length, dim_input, dim_attn=dim_attn, n_layer=n_layer, n_head=n_head)

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ
        W_PV = self.W_PV

        # faster to just use final token as the query, not whole context (we throw it away later)
        attn_arg = torch.transpose(xs, 1, 2) @ W_KQ @ xs[:, :, [-1]] / self.rho

        softmax_attn_arg = torch.softmax(attn_arg, dim=1)
        f_attn = W_PV @ xs @ softmax_attn_arg  # the residual stream term "+ xs" has been removed

        out = f_attn[:, :, -1]  # take dim_n output result at last token, for all batches
        return out


class TransformerModelV2noresOmitLast(TransformerModelV2):
    """
    See docstring TransformerModelV2
    """
    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__(context_length, dim_input, dim_attn=dim_attn, n_layer=n_layer, n_head=n_head)

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ
        W_PV = self.W_PV
        #rho = n_tokens

        xs_skip_last = xs[:, :, :-1]
        attn_arg = torch.transpose(xs_skip_last, 1, 2) @ W_KQ @ xs[:, :, [-1]] / self.rho

        # p7 Bartlett: "Softmax applied column-wise" (dim = data dim, not token dim)
        softmax_attn_arg = torch.softmax(attn_arg, dim=1)
        f_attn = W_PV @ xs_skip_last @ softmax_attn_arg  # the residual stream term "+ xs" has been removed

        out = f_attn[:, :, -1]  # take dim_n output result at last token, for all batches

        return out


class TransformerModelQKVnores(nn.Module):
    """
    Simplest model:
    - no positional encoding is used
    - same as V1 but now softmax in place of `linear` self-attention
    """
    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__()
        assert n_layer == 1
        assert n_head == 1
        assert dim_attn is None

        # attention matrices (need to split by head...)
        self.W_Q = weight_matrix(dim_input,  dim_input, mode='default')
        self.W_K = weight_matrix(dim_input,  dim_input, mode='default')
        self.W_V = weight_matrix(dim_input,  dim_input, mode='default')

        self.rho = 1.0

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        ####batchsz, n_dim, n_tokens = xs.size()

        Q = self.W_Q @ xs[:, :, [-1]]
        K = self.W_K @ xs
        V = self.W_V @ xs

        #QK_d = (Q @ K.T) / self.rho
        KQ_d = torch.transpose(K, 1, 2) @ Q / self.rho  # this is tensor-argument of softmax attention
        prob = torch.softmax(KQ_d, dim=1)
        attention = V @ prob

        out = attention[:, :, -1]  # take dim_n output result at last token, for all batches

        return out


def load_model_from_rundir(dir_run, epoch_int=None):
    """
    Step 0: assume rundir has structure used in io_dict above
    Step 1: read dim_n and context length from runinfo.txt
    Step 2: load model at particular epoch from model_checkpoints (if epoch unspecified, load model_end.pth)
    """
    # load runinfo settings
    runinfo_dict = load_runinfo_from_rundir(dir_run)
    dim_n = runinfo_dict['dim_n']
    context_length = runinfo_dict['context_len']
    nn_model_str = runinfo_dict['model']
    epochs = runinfo_dict['epochs']

    if epoch_int is not None:
        model_fpath = dir_run + os.sep + 'model_checkpoints' + os.sep + 'model_e%d.pth' % epoch_int
    else:
        model_fpath = dir_run + os.sep + 'model_checkpoints' + os.sep + 'model_final.pth'

    # nn_model_str is a string like 'TransformerModelV1' or its alias 'TV1' (short form)
    if nn_model_str in MODEL_CLASS_ALIAS_TO_STR.keys():
        nn_model_str = MODEL_CLASS_ALIAS_TO_STR[nn_model_str]
        nn_class = MODEL_CLASS_FROM_STR[nn_model_str]['class']
    else:
        nn_class = MODEL_CLASS_FROM_STR[nn_model_str]['class']

    net = nn_class(context_length, dim_n)   # TODO this currently assumes all models have same two inputs
    print('loading model at:', model_fpath, '...')
    net.load_state_dict(torch.load(model_fpath))
    net.eval()

    return net


def load_modeltype_from_fname(fname, dir_models=DIR_MODELS, model_params=None):
    """
    Trained model checkpoint files are assumed to follow a certain naming convention, and are placed in DIR_MODELS
        e.g. models\basicnetmult_chkpt_e240_L100_n128.pth

    If model_params is None, then the model params will be inferred from the filename itself
    """
    print('\nLoading model checkpoint from file...')
    fpath = dir_models + os.sep + fname
    print('...', fpath)

    if model_params is None:
        print('model_params is None; inferring class init from filename...')
        model_type = fname.split('_')[0]
        context_length = int(fname.split('_L')[1].split('_')[0])
        dim_input = int(fname.split('_n')[1].split('_')[0])
    else:
        model_type = model_params['nn_model']
        context_length = int(model_params['context_length'])
        dim_input = int(model_params['dim_input'])

    # nn_model_str is a string like 'TransformerModelV1' or its alias 'TV1' (short form)
    if model_type in MODEL_CLASS_ALIAS_TO_STR.keys():
        nn_model_str = MODEL_CLASS_ALIAS_TO_STR[model_type]
        nn_class = MODEL_CLASS_FROM_STR[nn_model_str]['class']
    else:
        nn_class = MODEL_CLASS_FROM_STR[model_type]['class']

    print('class:', model_type,
          '\n\tcontext_length=%d, dim_input=%d' % (context_length, dim_input))
    net = nn_class(context_length, dim_input)
    print('loading weights from fpath:', fpath)
    net.load_state_dict(torch.load(fpath, weights_only=True))  # avoid FutureWarning in latest Torch ~2.5

    return net, model_type, context_length, dim_input


def count_parameters(model):
    """
    Use: print parameters of torch nn.Module in nice manner
    From: https://stackoverflow.com/questions/67546610/pretty-print-list-without-module-in-an-ascii-table

    Table is just a list of lists
    """
    def pretty_print(table, ch1="-", ch2="|", ch3="+"):
        if len(table) == 0:
            return
        max_lengths = [
            max(column)
            for column in zip(*[[len(cell) for cell in row] for row in table])
        ]
        for row in table:
            print(ch3.join(["", *[ch1 * l for l in max_lengths], ""]))
            print(
                ch2.join(
                    [
                        "",
                        *[
                            ("{:<" + str(l) + "}").format(c)
                            for l, c in zip(max_lengths, row)
                        ],
                        "",
                    ]
                )
            )
        print(ch3.join(["", *[ch1 * l for l in max_lengths], ""]))

    total_params = 0
    table = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.append([name, str(params)])
        total_params += params
    print("(Modules | Parameters)")
    pretty_print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


MODEL_CLASS_FROM_STR = {
    'TransformerModelV1':
        {'class': TransformerModelV1, 'alias': 'TV1'},
    'TransformerModelV1nores':
        {'class': TransformerModelV1nores, 'alias': 'TV1nr'},
    'TransformerModelV1noresForceDiag':
        {'class': TransformerModelV1noresForceDiag, 'alias': 'TV1nrFD'},
    'TransformerModelV1noresOmitLast':
        {'class': TransformerModelV1noresOmitLast, 'alias': 'TV1nrOL'},
    'TransformerModelV1noresForceDiagAndOmitLast':
        {'class': TransformerModelV1noresForceDiagAndOmitLast, 'alias': 'TV1nrFDOL'},
    'TransformerModelV2':
        {'class': TransformerModelV2, 'alias': 'TV2'},
    'TransformerModelV2nores':
        {'class': TransformerModelV2nores, 'alias': 'TV2nr'},
    'TransformerModelV2noresOmitLast':
        {'class': TransformerModelV2noresOmitLast, 'alias': 'TV2nrOL'},
    'TransformerModelQKVnores':
        {'class': TransformerModelQKVnores, 'alias': 'TQKVnr'},
}
# define companion dict mapping alias to class string
MODEL_CLASS_ALIAS_TO_STR = {v['alias']: k for k, v in MODEL_CLASS_FROM_STR.items()}


if __name__ == '__main__':
    # sequence: three vectors from R^2
    sample_sequence = np.array([[
        [1, 1, 1],
        [2, 3, 4]
    ]]).astype('float32')

    print('Prep sample input for each model:')
    print('=' * 20)
    sample_input = torch.tensor(sample_sequence)  # add a batch dimension to front
    print('sample_input.size()', sample_input.size())
    batchsz, n_dim, n_tokens = sample_input.size()

    # Demonstrate parameter count table and sample outputs
    print('\nModel (sequence): ReturnLastToken()')
    model = ReturnLastToken()
    count_parameters(model)
    print('sample_output:', model(sample_input))

    print(sample_input)
    print(torch.transpose(sample_input, 1,2))

    print('\nModel (sequence): TransformerModelV1()')
    model = TransformerModelV1(n_tokens, n_dim)
    count_parameters(model)
    print('sample_output:', model(sample_input))

    print('\nModel (sequence): TransformerModelV2()')
    model = TransformerModelV2(n_tokens, n_dim)
    count_parameters(model)
    print('sample_output:', model(sample_input))

    # ========= scrap
    print('\n========= scrap')
    # emulate forwards pass of linear self-attention (Bartlett 2023, Eq. 3)
    W_PV = torch.randn(2, 2)
    W_KQ = torch.randn(2, 2)
    rho = n_tokens

    print(type(W_PV), W_PV.dtype)
    print(type(W_KQ), W_KQ.dtype)
    print(type(sample_input), sample_input.dtype)

    attn_arg = torch.transpose(sample_input, 1,2) @ W_KQ @ sample_input / rho
    f_attn = sample_input + W_PV @ sample_input @ attn_arg

    print(f_attn)
    out = f_attn[:, :, -1]  # take dim_n output result at last token, for all batches
    print(out, out.size())
    print(out.flatten(), out.flatten().size())

    print('\nSynthesize train/test data for sequence models')
    print('='*20)
    x_train, y_train, x_test, y_test, _, _ = data_train_test_split_linear(
        context_len=100,
        dim_n=128,
        num_W_in_dataset=1000,
        context_examples_per_W=1,
        test_ratio=0.2,
        verbose=True,
        seed=0)
    print('x_train.shape:', x_train.shape)

    print('\nSample input -> output for ReturnLastToken() sequence model')
    print('='*20)
    model = ReturnLastToken()

    single_batch = x_train[0, :, :].unsqueeze(0)  # add back trivial batch dim
    print('\tsingle_batch tensor.size:', single_batch.shape)
    out_from_single_batch = model(single_batch)
    print('\tout_from_single_batch tensor.size:', out_from_single_batch.shape)

    full_batch = x_train[:, :, :]
    print('\n\tfull_batch tensor.size:', full_batch.shape)
    out_from_full_batch = model(full_batch)
    print('\tout_from_full_batch tensor.size:', out_from_full_batch.shape)

    print('\nSample input -> output for TransformerModelV1() sequence model')
    print('='*20)
    model = TransformerModelV1(100, 128)

    single_batch = x_train[0, :, :].unsqueeze(0)
    print('\tsingle_batch tensor.size:', single_batch.shape)
    out_from_single_batch = model(single_batch)
    print('\tout_from_single_batch tensor.size:', out_from_single_batch.shape)

    full_batch = x_train[:, :, :]
    print('\n\tfull_batch tensor.size:', full_batch.shape)
    out_from_full_batch = model(full_batch)
    print('\tout_from_full_batch tensor.size:', out_from_full_batch.shape)

    print('\nSample input -> output for TransformerModelV2() sequence model')
    print('='*20)
    model = TransformerModelV2(100, 128)

    single_batch = x_train[0, :, :].unsqueeze(0)
    print('\tsingle_batch tensor.size:', single_batch.shape)
    out_from_single_batch = model(single_batch)
    print('\tout_from_single_batch tensor.size:', out_from_single_batch.shape)

    full_batch = x_train[:, :, :]
    print('\n\tfull_batch tensor.size:', full_batch.shape)
    out_from_full_batch = model(full_batch)
    print('\tout_from_full_batch tensor.size:', out_from_full_batch.shape)
