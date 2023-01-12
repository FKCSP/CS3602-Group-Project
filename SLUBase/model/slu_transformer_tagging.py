#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch.autograd import Variable
import math
from einops import rearrange
from functools import partial

def exists(val):
    return val is not None

def empty(tensor):
    return tensor.numel() == 0

def default(val, d):
    return val if exists(val) else d

class TagTransformer(nn.Module):

    def __init__(self, config):
        super(TagTransformer, self).__init__()
        self.config = config
        self.cell = config.encoder_cell
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.rnn = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(config.embed_size, config.embed_size)
        self.transformer = Transformer(config.embed_size, config.num_layer, 2, config.batch_size)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.norm = nn.LayerNorm(config.embed_size)
        self.output_layer = TaggingFNNDecoder(config.hidden_size+config.embed_size, config.num_tags, config.tag_pad_idx)

    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths

        embed = self.word_embed(input_ids)
        trans_input = self.fc(embed)
        #trans_input = self.dropout_layer(trans_input)
        packed_inputs = rnn_utils.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=True)
        packed_rnns_out, h_t_c_t = self.rnn(packed_inputs)  # bsize x seqlen x dim
        rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnns_out, batch_first=True)
        rnn_out = self.dropout_layer(rnn_out)
        trans_out = self.transformer(trans_input)
        trans_out = self.norm(trans_out)
        hiddens = torch.cat((rnn_out,trans_out),dim=2)
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids)

        return tag_output

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        prob, loss = self.forward(batch)
        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        return predictions, labels, loss.cpu().item()


class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return prob

class PreLayerNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):

        return self.fn(self.norm(x), **kwargs)

class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim = -1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.chunks == 1:
            return self.fn(x, **kwargs)
        chunks = x.chunk(self.chunks, dim = self.dim)
        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim = self.dim)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = None, glu = False):
        super().__init__()
        activation = default(activation, nn.GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x

class FullAttention(nn.Module):
    def __init__(self,
                 dim,
                 heads = 8,
                 dim_head = 64,
                 dropout = 0.,
                 qkv_bias = False
                ):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        
        self.heads = heads
        self.dim_head = dim_head
        
        self.to_q = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x) # [B, N, H*D]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v)) # [B, H, N, D]
        
        scaled_attention_logits = torch.einsum("bhxd,bhyd -> bhxy", q, k) / torch.sqrt(torch.tensor(self.dim_head, dtype=torch.float32))
        attention_weights = F.softmax(scaled_attention_logits, dim=-1) # [B, H, N, N]
        
        attn_output = torch.einsum("bhnx,bhxd -> bhnd", attention_weights, v) # [B, H, N, D]
        out = rearrange(attn_output, 'b h n d -> b n (h d)', h = h) # [B, N, H*D]
        out = self.to_out(out)
        
        return self.dropout(out)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, batch, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        print('pe',pe.shape)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        self.batch = batch

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],requires_grad=False)
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 batch_size,
                 dim_head = 64,
                 ff_chunks = 1,
                 ff_mult = 4,
                 ff_glu = False,
                 ff_dropout = 0.,
                 attn_dropout = 0.,
                 qkv_bias = True,
                ):
        super().__init__()
        self.depth = depth
        self.attns = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.pos_emb = PositionalEncoding(dim, 0.2, batch_size)

        wrapper_fn = partial(PreLayerNorm, dim)
        for _ in range(depth):
            self.attns.append(wrapper_fn(FullAttention(dim, heads, dim_head, attn_dropout, qkv_bias)))
            self.ffns.append(wrapper_fn(Chunk(ff_chunks, FeedForward(dim, mult=ff_mult, dropout=ff_dropout, glu=ff_glu), along_dim=1)))
    
    def forward(self, x):
        x += self.pos_emb(x)
        for i in range(self.depth):
            x = x + self.attns[i](x) # residual link
            x = x + self.ffns[i](x) # residual link
        return x

class TransformerLM_mse(nn.Module):
    def __init__(self,
                 num_tokens,
                 max_seq_len,
                 dim,
                 depth,
                 heads,
                 dim_head=64,
                 ff_chunks=1,
                 ff_mult=4,
                 ff_glu=False,
                 ff_dropout=0.,
                 emb_dropout=0.,
                 attn_dropout=0.,
                 g2v_position_emb=False,
                 qkv_bias=True,
                 ):
        super(TransformerLM_mse, self).__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)

        if g2v_position_emb:
            self.pos_emb = Gene2VecPositionalEmbedding(dim, max_seq_len)
            print("pos_emb Gene2Vec", self.pos_emb)
        else:
            self.pos_emb = RandomPositionalEmbedding(dim, max_seq_len)
            print("pos_emb no ", self.pos_emb)

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, ff_chunks, ff_mult, ff_glu, ff_dropout,
                                       attn_dropout, qkv_bias)
        self.norm = nn.LayerNorm(dim)
        self.to_out = nn.Linear(dim, num_tokens)
        self.to_final = nn.Linear(num_tokens, 1)

    def forward(self, x):
        b, n, device = *x.shape, x.device
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'

        # token and positional embedding
        x = self.token_emb(x)
        x += self.pos_emb(x)

        x = self.dropout(x)  # embedding dropout

        x = self.transformer(x)  # get encodings [B, N, D]

        # layernorm and to logits
        x = self.norm(x)
        x = self.to_out(x)  # [B, N, C]

        if exists(self.to_final):
            x = self.to_final(x)
            return x.squeeze(2)  # torch.Size([8, 13418])
        else:
            return x

        return x