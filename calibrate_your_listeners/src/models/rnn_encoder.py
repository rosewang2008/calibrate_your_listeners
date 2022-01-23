import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.utils.rnn as rnn_utils

class RNNEncoder(nn.Module):
    """
    RNN Encoder - takes in onehot representations of tokens, rather than numeric
    """
    def __init__(self, embedding_module, is_old=False, hidden_size=100, dropout=0.):
        super(RNNEncoder, self).__init__()
        self.embedding = embedding_module
        self.embedding_dim = embedding_module.embedding_dim
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.embedding_dim, hidden_size, dropout=dropout, num_layers=2)

        self.vocab_size = self.embedding.num_embeddings
        self._is_old = is_old

    def forward(self, seq, length, used_as_internal_listener=False):
        """Performs g(u), embeds the utterances.
        :param seq: Tensor of size (max_seq_length, batch_size, vocab_size).
            max_sequence_lenth is typically 40.
            batch_size is typically 32.
            vocab_size is typically 1201 for colors.
        :param length: Tensor of length batch_size. Specifies the length
            of each sequence in the batch, ie. elements are <= max_seq_length.
        :returns: new hidden state, h_t. shape is (batch_size, hidden_size)
            hidden_size is typically 100.
        """
        if not used_as_internal_listener and not self._is_old:
            seq = seq['input_ids']
            seq = F.one_hot(seq,
                    num_classes=self.vocab_size).float()
        batch_size = seq.size(0)

        if batch_size > 1:
            sorted_lengths, sorted_idx = torch.sort(length, descending=True)
            seq = seq[sorted_idx]

        # reorder from (B,L,D) to (L,B,D)
        seq = seq.transpose(0, 1)

        # embed your sequences
        embed_seq = seq.cuda() @ self.embedding.weight
        # embed_seq = self.dropout(embed_seq)

        # TODO Bug with length = 0; length should always be >= 1, because of
        # mandatory EOS token
        # sorted_lengths += 1

        packed = rnn_utils.pack_padded_sequence(
            embed_seq,
            sorted_lengths.data.tolist() if batch_size > 1 else length.data.tolist())

        _, hidden = self.gru(packed)
        hidden = hidden[-1, ...]

        if batch_size > 1:
            _, reversed_idx = torch.sort(sorted_idx)
            hidden = hidden[reversed_idx]

        return hidden
