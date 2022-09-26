import hydra
from pl_bolts.models.regression import LinearRegression
import pytorch_lightning as pl
from pl_bolts.datamodules import SklearnDataModule
from sklearn.datasets import load_diabetes
from pytorch_lightning.plugins.environments import SLURMEnvironment
import numpy as np
import time
from pytorch_lightning.accelerators import CUDAAccelerator
from pytorch_lightning.utilities import device_parser
from pytorch_lightning.strategies.launchers.multiprocessing import _is_forking_disabled
import multiprocessing
import torch

@hydra.main(config_name="sigterm", config_path="config")
def main_except(cfg):

    time.sleep(2)


    print("multi", flush=True)
    with multiprocessing.get_context("fork").Pool(1) as pool:
        time.sleep(2)
        print("apply", flush=True)
        pool.apply(torch.cuda.device_count)
        time.sleep(2)    

    print('done', flush=True)
    time.sleep(2)

if __name__ == "__main__":
    main_except()
