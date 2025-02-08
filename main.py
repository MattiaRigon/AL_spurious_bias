import logging
from dataclasses import asdict

import hydra
import wandb
from omegaconf import OmegaConf

from conf import job

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
    main()
