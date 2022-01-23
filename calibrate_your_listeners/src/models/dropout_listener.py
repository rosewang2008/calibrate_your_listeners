import numpy as np
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from calibrate_your_listeners.src.models import listener
from calibrate_your_listeners.src.models import (
    vision,
    rnn_encoder,
)

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2

MAX_SEQ_LEN=10
EPS=1e-5


class DropoutListener(listener.Listener): # L_0
    def __init__(self, config, max_seq_len=MAX_SEQ_LEN):

        self.dropout_rate = config.model_params.dropout_rate

        super(DropoutListener, self).__init__(
             config=config,
             max_seq_len=max_seq_len)

    def init_lang_feature_model(self):
        self.lang_model = rnn_encoder.RNNEncoder(self.embedding, dropout=self.dropout_rate) # g

    def init_image_feature_model(self):
        self.feat_model = vision.Conv4(dropout=self.dropout_rate)

    @property
    def is_old(self):
        return self._is_old

    def forward(self, feats, lang, lang_length,
                num_passes=1,
                average=False, used_as_internal_listener=False):

        vision_pass = []
        lang_pass = []

        for _ in range(num_passes):
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

            vision_pass.append(feats_emb.unsqueeze(0))
            lang_pass.append(lang_emb.unsqueeze(0))
        # Compute dot products, shape (batch_size, num_images in reference game)
        # L_0(t|I, u) ~ exp (f_L(I_t)^T g(u))
        # scores = F.softmax(torch.einsum('ijh,ih->ij', (feats_emb, lang_emb)))
        v = torch.cat(vision_pass, dim=0)
        l = torch.cat(lang_pass, dim=0)
        emb_var_loss = l.var(0).sum()/(v.shape[1] * num_passes)
        scores = F.softmax(torch.einsum('ijh,ih->ij', (v.mean(0), l.mean(0))))
        return scores, emb_var_loss
