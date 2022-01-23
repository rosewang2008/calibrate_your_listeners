import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2Tokenizer

from calibrate_your_listeners.src.models import (
    vision,
)
from calibrate_your_listeners import constants




def to_onehot(y, n=3):
    y_onehot = torch.zeros(y.shape[0], n).to(y.device)
    y_onehot.scatter_(1, y.view(-1, 1), 1)
    return y_onehot

class Speaker(nn.Module): # L_0
    def __init__(self, config):

        super(Speaker, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self._is_old = (self.config.model_params.vocab == "shapeworld")
        self._max_seq_len = constants.MAX_SEQ_LEN

        self.set_vocab()
        self.initialize_modules()

    def set_vocab(self):
        if self._is_old:
            self.vocab_size = self.config.dataset_params.num_shapeworld_tokens
            self._set_tokens()
        else:
            self._tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self._set_tokens()
            self.vocab_size = self._tokenizer.vocab_size

    def initialize_modules(self):
        self.embedding = nn.Embedding(self.vocab_size, 50) # embedding_module

        self.init_lang_model()
        self.init_image_feature_model()

        self.image_feat_size = self.feat_model.final_feat_dim
        self.n_images = self.config.dataset_params.n_images
        self.imgFeat2hidden = nn.Linear(
            self.n_images * (self.image_feat_size + 1),
            self.hidden_size)

    def init_lang_model(self):
        self.hidden_size = self.config.model_params.hidden_size
        self.gru = nn.GRU(
            self.embedding.embedding_dim, self.hidden_size)
        self.hidden2vocab = nn.Linear(self.hidden_size, self.vocab_size)

    def init_image_feature_model(self):
        self.feat_model = vision.Conv4() # f_L(I_t)

    @property
    def is_old(self):
        return self._is_old

    def _set_tokens(self):
        if self.is_old:
            self._start_token = constants.SOS_IDX
            self._end_token = constants.EOS_IDX
        else:
            self._start_token = None
            self._end_token = self._tokenizer.eos_token_id
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._torch_end_token = torch.tensor(self._end_token).to(self.device)

    def get_trainable_parameters(self, freeze_mode):
        return self.parameters()


    def embed_features(self, feats, targets):
        batch_size = feats.shape[0]
        n_obj = feats.shape[1]
        rest = feats.shape[2:]
        feats_flat = feats.view(batch_size * n_obj, *rest)
        feats_emb_flat = self.feat_model(feats_flat)
        feats_emb = feats_emb_flat.unsqueeze(1).view(batch_size, n_obj, -1)
        # Add targets
        targets_onehot = to_onehot(targets)
        feats_and_targets = torch.cat((feats_emb, targets_onehot.unsqueeze(2)), 2)
        ft_concat = feats_and_targets.view(batch_size, -1)
        return ft_concat

    def forward(self, feats, targets, activation='gumbel', tau=1.0, length_penalty=False):

        batch_size = feats.size(0)
        feats_emb = self.embed_features(feats, targets)

        # initialize hidden states using image features
        states = self.imgFeat2hidden(feats_emb)
        states = states.unsqueeze(0)

        # This contains are series of sampled onehot vectors
        lang = []
        lang_prob = None
        if length_penalty:
            eos_prob = []

        # And vector lengths
        lang_length = np.ones(batch_size, dtype=np.int64)
        done_sampling = np.array([False for _ in range(batch_size)])

        # first input is SOS token
        # (batch_size, n_vocab)
        inputs_onehot = torch.zeros(batch_size, self.vocab_size, device=feats.device)
        if self._is_old:
            inputs_onehot[:, constants.SOS_IDX] = 1.0

        # No start token for GPT - leave inputs as onehot

        # (batch_size, len, n_vocab)
        inputs_onehot = inputs_onehot.unsqueeze(1)

        # Add SOS to lang
        lang.append(inputs_onehot)

        # (B,L,D) to (L,B,D)
        inputs_onehot = inputs_onehot.transpose(0, 1)

        # compute embeddings
        # (1, batch_size, n_vocab) X (n_vocab, h) -> (1, batch_size, h)
        inputs = inputs_onehot @ self.embedding.weight
        if self._is_old:
            max_len = self._max_seq_len - 2
        else:
            max_len = self._max_seq_len - 1

        for i in range(max_len):  # Have room for EOS if never sampled
            # FIXME: This is inefficient since I do sampling even if we've
            # finished generating language.
            if all(done_sampling):
                break

            outputs, states = self.gru(inputs, states)  # outputs: (L=1,B,H)
            outputs = outputs.squeeze(0)                # outputs: (B,H)
            outputs = self.hidden2vocab(outputs)       # outputs: (B,V)

            if activation=='gumbel':
                predicted_onehot = F.gumbel_softmax(outputs, tau=tau, hard=True)
            else:
                raise NotImplementedError(activation)

            # Add to lang
            lang.append(predicted_onehot.unsqueeze(1))
            if length_penalty:
                idx_prob = F.log_softmax(outputs, dim = 1)
                eos_prob.append(idx_prob[:, self._end_token])

            # Update language lengths
            lang_length += ~done_sampling
            done_sampling = np.logical_or(
                done_sampling,
                (predicted_onehot[:, self._end_token] == 1.0).cpu().numpy())
            # assert activation in {'gumbel', 'multinomial'}, "check activation either gumbel or multinom"

            # (1, batch_size, n_vocab) X (n_vocab, h) -> (1, batch_size, h)
            inputs = (predicted_onehot.unsqueeze(0)) @ self.embedding.weight

        # Add EOS if we've never sampled it
        eos_onehot = torch.zeros(batch_size, 1, self.vocab_size, device=feats.device)
        eos_onehot[:, 0, self._end_token] = 1.0
        lang.append(eos_onehot)

        # Cut off the rest of the sentences
        lang_length += (~done_sampling)

        # Cat language tensors (batch_size, max_seq_length, vocab_size)
        # skip first element b/c it's just 0s
        # no SOS token for GPT
        if self._is_old:
            lang_tensor = torch.cat(lang, 1)
        else:
            lang_tensor = torch.cat(lang[1:], 1)
            lang_length -= 1

        for i in range(lang_tensor.shape[0]):
            lang_tensor[i, lang_length[i]:] = 0

        # Trim max length
        max_lang_len = lang_length.max()
        lang_tensor = lang_tensor[:, :max_lang_len, :]

        if length_penalty:
            # eos prob -> eos loss
            eos_prob = torch.stack(eos_prob, dim = 1)
            for i in range(eos_prob.shape[0]):
                r_len = torch.arange(1,eos_prob.shape[1]+1,dtype=torch.float32)
                eos_prob[i] = eos_prob[i]*r_len.to(eos_prob.device)
                eos_prob[i, lang_length[i]:] = 0
            eos_loss = -eos_prob
            eos_loss = eos_loss.sum(1)/torch.tensor(
                lang_length,dtype=torch.float32, device=eos_loss.device)
            eos_loss = eos_loss.mean()
        else:
            eos_loss = 0

        # Sum up log probabilities of samples
        lang_length = torch.Tensor(lang_length)
        return lang_tensor, lang_length, eos_loss # , lang_prob
