import hydra
from pl_bolts.models.regression import LinearRegression
import pytorch_lightning as pl
from pl_bolts.datamodules import SklearnDataModule
from sklearn.datasets import load_diabetes
from pytorch_lightning.plugins.environments import SLURMEnvironment
import signal

@hydra.main(config_name="sigterm", config_path="config")
def main_except(cfg):
    import os
    del os.environ["SLURM_NTASKS"]
    del os.environ["SLURM_JOB_NAME"]

    X, y = load_diabetes(return_X_y=True)
    loaders = SklearnDataModule(X, y)

    model = LinearRegression(input_dim=10)
    #signal.signal(signal.SIGUSR2, signal.SIG_DFL)
    trainer = pl.Trainer(accelerator="cpu",  # error on cpu and gpu
                         plugins=[SLURMEnvironment(auto_requeue=False)],
                         enable_progress_bar=False)
    trainer.fit(model, train_dataloaders=loaders.train_dataloader(), val_dataloaders=loaders.val_dataloader())


if __name__ == "__main__":
    main_except()
