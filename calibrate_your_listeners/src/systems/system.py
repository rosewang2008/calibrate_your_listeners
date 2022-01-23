import torch
from torch.utils.data import DataLoader
from calibrate_your_listeners import constants
from calibrate_your_listeners.src.systems import utils
import pytorch_lightning as pl
import wandb
torch.autograd.set_detect_anomaly(True)


class BasicSystem(pl.LightningModule):
    """
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.set_datasets()
        self.set_models()

        self.num_train_step = 0


    def set_datasets(self):
        self.train_dataset = constants.NAME2DATASETS[self.config.dataset_params.name](
            train=True,
            config=self.config,
        )

        self.val_dataset = constants.NAME2DATASETS[self.config.dataset_params.name](
            train=False,
            config=self.config,
        )

        self.test_dataset = constants.NAME2DATASETS[self.config.dataset_params.name](
            train=False,
            config=self.config,
        )

    def set_models(self):
        self.model = constants.NAME2MODELS[self.config.model_params.name](
            config=self.config,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.optim_params.learning_rate)
        return [optimizer], []

    def get_losses_for_batch(self, batch, batch_idx):
        raise NotImplementedError()

    def _convert_results_to_floats(self, result):
        return {k: v.item() for k, v in result.items()}

    def _add_wandb_category(self, result, category):
        return {f"{category}/{k}": v for k, v in result.items()}

    def log_results(self, result, category):
        wandb_result = self._convert_results_to_floats(result)
        wandb_result = self._add_wandb_category(wandb_result, category=category)
        wandb_result['epoch'] = self.trainer.current_epoch
        wandb.log(wandb_result)

    def training_step(self, batch, batch_idx):
        result = self.get_losses_for_batch(batch, batch_idx)
        loss = result['loss']
        self.log_results(result=result, category="train")
        return loss

    def validation_step(self, batch, batch_idx):
        result = self.get_losses_for_batch(batch, batch_idx)
        loss = result['loss']

        self.log_results(result=result, category="val")
        return loss

    def test_step(self, batch, batch_idx):
        result = self.get_losses_for_batch(batch, batch_idx)
        loss = result['loss']
        self.log_results(result=result, category="test")
        return loss

    def train_dataloader(self):
        return utils.create_dataloader(self.train_dataset, self.config)

    def test_dataloader(self):
        return utils.create_dataloader(self.test_dataset, self.config, shuffle=False)

    def val_dataloader(self):
        return utils.create_dataloader(self.val_dataset, self.config, shuffle=False)
