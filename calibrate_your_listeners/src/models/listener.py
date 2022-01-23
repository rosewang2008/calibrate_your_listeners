import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2Tokenizer

from calibrate_your_listeners.src.models import (
    vision,
    rnn_encoder,
)
from calibrate_your_listeners import constants

class Listener(nn.Module): # L_0
    def __init__(self, config, max_seq_len=constants.MAX_SEQ_LEN):

        super(Listener, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self._is_old = (self.config.model_params.vocab == "shapeworld")
        self._max_seq_len = max_seq_len

        self.set_vocab()
        self.initialize_modules()


    def set_vocab(self):
        if self._is_old:
            self.vocab_size = self.config.dataset_params.num_shapeworld_tokens
        else:
            self._tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self._set_tokens()
            self.vocab_size = self._tokenizer.vocab_size

    def initialize_modules(self):
        self.embedding = nn.Embedding(self.vocab_size, 50) # embedding_module
        self.init_lang_feature_model()
        self.init_image_feature_model()
        self.image_feat_size = self.feat_model.final_feat_dim

        self.image2Joint = None
        self.lang2Joint = nn.Linear(self.lang_model.hidden_size, self.image_feat_size, bias=False)

    def init_lang_feature_model(self):
        self.lang_model = rnn_encoder.RNNEncoder(
            self.embedding, is_old=self._is_old) # g

    def init_image_feature_model(self):
        self.feat_model = vision.Conv4() # f_L(I_t)

    @property
    def is_old(self):
        return self._is_old

    def tokenize(self, utterances):
        encoded_input = self._tokenizer(
            utterances,
            padding=True,
            truncation=True,
            max_length=self._max_seq_len-1,
            return_tensors="pt")
        # pad
        seq_length = encoded_input['input_ids'].shape[1]
        eos_input_ids = torch.tensor([
            self._end_token for _ in range(self._max_seq_len-seq_length)]).unsqueeze(0)
        eos_attention = torch.tensor([0 for _ in range(self._max_seq_len-seq_length)]).unsqueeze(0)
        # Add an EOS token at the very end if it doesn't already exist
        # and add attention to ignore the EOS tokens
        # batch_size x 1
        # eos_input_ids = torch.tensor([self._end_token for _ in range(batch_size)]).unsqueeze(1)
        encoded_input['input_ids'] = torch.cat((encoded_input['input_ids'],
                                                eos_input_ids), dim=1)
        encoded_input['attention_mask'] = torch.cat((encoded_input['attention_mask'],
                                                eos_attention), dim=1)
        encoded_input = {k : v.squeeze(0) for k, v in encoded_input.items()}
        return encoded_input# .to(self.device)

    def _set_tokens(self):
        # Adding padding token to GPT tokenizer
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._start_token = None
        self._end_token = self._tokenizer.eos_token_id
        self._torch_end_token = torch.tensor(self._end_token).to(self.device)

    def get_length(self, lang):
        if self.is_old:
            return torch.tensor([np.count_nonzero(t) for t in lang.argmax(-1).cpu()], dtype=np.int)
        else:
            return torch.tensor([
                np.count_nonzero(t) for t in lang['attention_mask'].cpu()], dtype=np.int)

    def get_trainable_parameters(self, freeze_mode):
        return self.parameters()

    def embed_features(self, feats):
        batch_size = feats.shape[0]
        n_obj = feats.shape[1]
        rest = feats.shape[2:]
        feats_flat = feats.reshape(batch_size * n_obj, *rest)
        feats_emb_flat = self.feat_model(feats_flat)
        feats_emb = feats_emb_flat.unsqueeze(1).view(batch_size, n_obj, -1)
        return feats_emb

    def forward(self, feats, lang, lang_length, average=False, used_as_internal_listener=False):
        """Calculates: L_0(t|I, u) ~ exp (f_L(I_t)^T g(u)) [from paper].
        :param feats: tensor of shape (batch_size, num_imgs in reference game,
            num_img_channels, img_width, img_height).
            Represents the images in each reference game, I_t where
            0 <= t < num_imgs in reference game.
        :param lang: tensor of shape (batch_size, max_seq_length, vocab_size).
            Represents the speaker utterances, u.
        :param lang_length: tensor of shape (batch_size,).
            Represents the actual length of each sequence.
        :returns: softmax of listener's beliefs over images in reference game.
        """
        # Embed features, f_L(I_t)
        feats_emb = self.embed_features(feats)
        # Image -> joint space if using a small space
        if self.image2Joint is not None:
            feats_emb = self.image2Joint(feats_emb)

        # Embed language, g(u)
        lang_emb = self.lang_model(lang, lang_length,
                                    used_as_internal_listener) # 32, 40, 15 (batch, max_sentence, vocab_size)
        # lang -> joint space
        lang_emb = self.lang2Joint(lang_emb)

        # Compute dot products, shape (batch_size, num_images in reference game)
        # L_0(t|I, u) ~ exp (f_L(I_t)^T g(u))
        scores = F.softmax(torch.einsum('ijh,ih->ij', (feats_emb, lang_emb)))
        return scores, 0.0

