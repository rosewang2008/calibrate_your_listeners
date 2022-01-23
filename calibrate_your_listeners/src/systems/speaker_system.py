from calibrate_your_listeners.src.systems import system
from calibrate_your_listeners.src.models import (
    listener,
    dropout_listener,
    speaker,
)
from calibrate_your_listeners.src.objectives import (
    listener_scores,
    dropout_listener_scores,
)
from calibrate_your_listeners.src.systems import utils
from calibrate_your_listeners import constants
from pytorch_lightning.trainer.supporters import CombinedLoader

import os
import torch
import torch.nn as nn
import pandas as pd

class SpeakerSystem(system.BasicSystem):

    def __init__(self, config):
        super().__init__(config=config)

        self.post_model_init()

    def post_model_init(self):
        model = self.train_listeners[0]
        self.train_dataset.listener_tokenize_f=model.tokenize
        self.val_dataset.listener_tokenize_f=model.tokenize
        self.test_dataset.listener_tokenize_f=model.tokenize

    def freeze_model(self, model):
        for p in model.parameters():
            p.requires_grad = False

    def _load_listener(self, listener_type, vocab_type, listener_idx):
        if listener_type == "normal":
            l0 = listener.Listener(config=self.config)
            l0_dir = constants.NORMAL_LISTENER_MODEL_DIR
            model_fname = os.path.join(
                l0_dir,
                f"sw_l0_{listener_idx}",
                "epoch=99-step=12499.ckpt")
            self.l0_scorer = listener_scores.ListenerScores
        elif listener_type == "ensemble":
            l0 = listener.Listener(config=self.config)
            l0_dir = constants.NORMAL_LISTENER_MODEL_DIR
            model_fname = os.path.join(
                l0_dir,
                f"ensemble_l0_si{listener_idx}",
                "epoch=99-step=12499.ckpt")
            self.l0_scorer = listener_scores.ListenerScores
        elif listener_type == "dropout":
            l0 = dropout_listener.DropoutListener(config=self.config)
            l0_dir = constants.DROPOUT_LISTENER_MODEL_DIR
            model_fname = os.path.join(
                l0_dir,
                f"dropout_l0_si{listener_idx}",
                "epoch=99-step=12499.ckpt")
            self.l0_scorer = dropout_listener_scores.DropoutListenerScores

        print(f'Loading listener from {model_fname}')
        state_dict = self._load_and_process_state_dict(model_fname)
        l0.load_state_dict(state_dict)

        # Keep dropout for speaker
        # if listener_type == "normal":
        #     l0.eval()

        return l0

    def _load_and_process_state_dict(self, model_fname):
        state_dict = torch.load(model_fname)['state_dict']
        new_state_dict = dict()

        for k, v in state_dict.items():
            key_ = 'model.'
            if k.startswith(key_):
                k = k[len(key_):]
            new_state_dict[k] = v
        return new_state_dict

    def load_listeners(self):
        self.train_listeners = []
        self.val_listeners = []
        # Training listeener
        for listener_idx in range(0, self.config.listener_params.ensemble_size):
            print('Loading training listener')
            print(f'Train idx: {listener_idx}')
            listener = self._load_listener(
                listener_type=self.config.listener_params.type,
                vocab_type=self.config.model_params.vocab,
                listener_idx=listener_idx+1 # idx start at 1 not 0
                )
            self.freeze_model(listener)
            self.train_listeners.append(listener)

        for listener_idx in range(self.config.listener_params.ensemble_size,
                                  2*self.config.listener_params.ensemble_size):
            # Val listeners
            print('Loading validation listener')
            # val_idx = (self.config.listener_params.val_idx
            #            if self.config.listener_params.val_idx else listener_idx)
            print(f'Val idx: {listener_idx}')
            listener = self._load_listener(
                listener_type=self.config.listener_params.type,
                vocab_type=self.config.model_params.vocab,
                listener_idx=listener_idx+1
                )
            self.freeze_model(listener)
            self.val_listeners.append(listener)
        print(f'A training listener arch: {self.train_listeners[0]}')
        print(f'A validation listener arch: {self.val_listeners[0]}')

    def load_speaker(self):
        self.model = speaker.Speaker(config=self.config)

    def set_models(self):
        num_tokens = len(self.train_dataset.vocab['w2i'].keys())
        self.config.dataset_params.num_shapeworld_tokens = num_tokens
        self.load_listeners()
        self.load_speaker()

    def _process_gt(self, gt):
        if isinstance(gt, dict):
            result = []
            for seq_id in range(gt['input_ids'].shape[0]):
                result.append(self.train_listeners[0]._tokenizer.decode(gt['input_ids'][seq_id]))
            return result
        else:
            return self.train_dataset.to_text(gt.argmax(-1))

    def construct_lang_table(self, lang, gt):
        data = []
        text_gts = self._process_gt(gt)
        if isinstance(gt, dict):
            lang = {'input_ids': lang.argmax(-1)}
        text_langs = self._process_gt(lang) # self.train_dataset.to_text(lang.argmax(-1))
        for l, gt in zip(text_langs, text_gts):
            data.append({
                'epoch': self.trainer.current_epoch,
                'speaker_utterance': l,
                'ground_truth': gt
            })
        return pd.DataFrame(data)

    def get_losses_for_batch(self, batch, batch_idx, which_listener):
        imgs, labels, utterances = (
            batch['imgs'].float(), batch['label'].argmax(-1).long(), batch['utterance'])
        lang, lang_length, loss = self.model(imgs, labels)

        listeners = self.train_listeners if which_listener == "train" else self.val_listeners

        l0_scorer = self.l0_scorer(
            listeners=listeners,
            imgs=imgs,
            lang=lang,
            lang_length=lang_length,
            config=self.config
        )
        avg_l0_scores = l0_scorer.get_average_l0_score()
        lis_pred = avg_l0_scores.argmax(1)
        loss = nn.CrossEntropyLoss()
        losses = loss(avg_l0_scores, labels)
        return {
            'loss': losses,
            'acc': (lis_pred == labels).float().mean(),
            # 'lang_table': wandb.Table(
            #     dataframe=self.construct_lang_table(lang=lang, gt=utterances)
            # )
        }

    def _convert_results_to_floats(self, result):
        results = dict()
        for k, v in result.items():
            if k == "lang_table":
                results[k] = v
            else:
                results[k] = v.item()
        return results

    def log_results(self, result, category):
        # if category == "train":
        #     # remove lang table
        #     result.pop('lang_table', None)
        super().log_results(result, category)

    def training_step(self, batch, batch_idx):
        result = self.get_losses_for_batch(batch, batch_idx, which_listener="train")
        loss = result['loss']
        self.log_results(result=result, category="train")
        return loss

    def test_step(self, batch, batch_idx):
        result = self.get_losses_for_batch(batch, batch_idx, which_listener="test")
        loss = result['loss']
        self.log_results(result=result, category="test")
        return loss

    def validation_step(self, batch, batch_idx):
        for setting in ["trainL0_trainD", "trainL0_valD", "valL0_trainD", "valL0_valD"]:
            which_listener = "train" if "trainL0" in setting else "val"
            result = self.get_losses_for_batch(
                batch[setting], batch_idx, which_listener=which_listener)
            loss = result['loss']
            self.log_results(result=result, category=setting)
        return loss

    def val_dataloader(self):
        # train L0 - train D
        # train L0 - val D
        # val L0 - train D
        # val L0 - val D
        loaders = {
            "trainL0_trainD": utils.create_dataloader(self.train_dataset, self.config, shuffle=False),
            "trainL0_valD": utils.create_dataloader(self.val_dataset, self.config, shuffle=False),
            "valL0_trainD": utils.create_dataloader(self.train_dataset, self.config, shuffle=False),
            "valL0_valD": utils.create_dataloader(self.val_dataset, self.config, shuffle=False),
        }
        combined_loaders = CombinedLoader(loaders, "max_size_cycle")
        return combined_loaders
