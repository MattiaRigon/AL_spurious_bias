import logging
from dataclasses import asdict
import os

import hydra
import numpy as np
import wandb
from omegaconf import OmegaConf

from conf import job
# from lang_sam import LangSAM

from hydra.core.global_hydra import GlobalHydra
from hydra.core.global_hydra import GlobalHydra
from PIL import Image
# from utils.singleton_langsam import LangSAMSingleton

to_container = OmegaConf.to_container
to_object = OmegaConf.to_object
to_dict = lambda cfg: asdict(to_object(cfg))  # noqa: E731


@hydra.main(version_base=None, config_path="conf")
def main(cfg: job.JobBase):
    job_name = cfg._metadata.object_type.__name__
    config = to_container(cfg, resolve=True)
    with wandb.init(**to_dict(cfg.wandb), job_type=job_name, config=config):
        logging.info(f"job_type={job_name}")
        output = to_object(cfg).run()
    return output



if __name__ == "__main__":

    # # load the model using the singleton pattern, before to change configuration
    # singleton = LangSAMSingleton()
    
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # model = singleton.get_model()
    
    main()
