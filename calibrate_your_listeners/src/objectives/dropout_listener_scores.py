import torch

from calibrate_your_listeners.src.objectives import listener_scores

class DropoutListenerScores(listener_scores.ListenerScores):

    def __init__(self, listeners, imgs, lang, lang_length, **kwargs):
        self.num_passes = kwargs['config'].training_params.num_dropout_passes
        super().__init__(
            listeners=listeners,
            imgs=imgs,
            lang=lang,
            lang_length=lang_length
        )

    def _calculate_listener_scores(self):
        lis_scores = []
        self.dropout_listener = self.listeners[0]
        for _ in range(self.num_passes):
            if self.dropout_listener.device != self.imgs.device:
                l0 = self.dropout_listener.to(self.imgs.device)
            lis_score, _ = self.dropout_listener(
                self.imgs, self.lang, self.lang_length,
                used_as_internal_listener=True
            )
            lis_scores.append(lis_score)
        lis_scores = torch.stack(lis_scores)
        return lis_scores

    def get_average_l0_score(self):
        return torch.mean(self.listener_scores, axis=0) # average across listeners
