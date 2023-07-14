# Copyright (C) 2023 Elif Cansu YILDIZ
# 
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
# 
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
# 
# You should have received a copy of the GNU General Public License along with
# this program; if not, see <http://www.gnu.org/licenses/>.

import warnings

from packaging import version

import optuna
from optuna.storages._cached_storage import _CachedStorage
from optuna.storages._rdb.storage import RDBStorage


# Define key names of `Trial.system_attrs`.
_PRUNED_KEY = "ddp_pl:pruned"
_EPOCH_KEY = "ddp_pl:epoch"


with optuna._imports.try_import() as _imports:
    import pytorch_lightning as pl
    from pytorch_lightning import LightningModule
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import Callback

if not _imports.is_successful():
    Callback = object  # type: ignore  # NOQA
    LightningModule = object  # type: ignore  # NOQA
    Trainer = object  # type: ignore  # NOQA


class PyTorchLightningPruningCallback(Callback):
    """PyTorch Lightning callback to prune unpromising trials.

    See `the example <https://github.com/optuna/optuna-examples/blob/
    main/pytorch/pytorch_lightning_simple.py>`__
    if you want to add a pruning callback which observes accuracy.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or
            ``val_acc``. The metrics are obtained from the returned dictionaries from e.g.
            ``pytorch_lightning.LightningModule.training_step`` or
            ``pytorch_lightning.LightningModule.validation_epoch_end`` and the names thus depend on
            how this dictionary is formatted.

    .. note::
        For the distributed data parallel training, the version of PyTorchLightning needs to be
        higher than or equal to v1.5.0. In addition, :class:`~optuna.study.Study` should be
        instantiated with RDB storage.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
        _imports.check()
        super().__init__()

        self.trial = trial
        self.monitor = monitor
        self.is_ddp_backend = False


    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:

        # When the trainer calls `on_validation_end` for sanity check,
        # do not call `trial.report` to avoid calling `trial.report` multiple times
        # at epoch 0. The related page is
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1391.
        if trainer.sanity_checking:
            return

        epoch = pl_module.current_epoch

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            message = (
                "The metric '{}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name.".format(self.monitor)
            )
            warnings.warn(message)
            return

        should_stop = False
        if trainer.is_global_zero:
            print("------------------------", current_score.item())
            self.trial.report(value=current_score.item(), step=epoch)
            should_stop = self.trial.should_prune()
        #should_stop = trainer.training_type_plugin.broadcast(should_stop) # deprecated
        should_stop = trainer.strategy.broadcast(should_stop)
        if not should_stop:
            return

        if not self.is_ddp_backend:
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)
        else:
            # Stop every DDP process if global rank 0 process decides to stop.
            trainer.should_stop = True
            if trainer.is_global_zero:
                self.trial.storage.set_trial_system_attr(self.trial._trial_id, _PRUNED_KEY, True)
                self.trial.storage.set_trial_system_attr(self.trial._trial_id, _EPOCH_KEY, epoch)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.is_ddp_backend:
            return

        # Because on_validation_end is executed in spawned processes,
        # _trial.report is necessary to update the memory in main process, not to update the RDB.
        _trial_id = self.trial._trial_id
        _study = self.trial.study
        _trial = _study._storage._backend.get_trial(_trial_id)  # type: ignore
        _trial_system_attrs = _study._storage.get_trial_system_attrs(_trial_id)
        is_pruned = _trial_system_attrs.get(_PRUNED_KEY)
        epoch = _trial_system_attrs.get(_EPOCH_KEY)
        intermediate_values = _trial.intermediate_values
        for step, value in intermediate_values.items():
            self.trial.report(value=value, step=step)

        if is_pruned:
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)