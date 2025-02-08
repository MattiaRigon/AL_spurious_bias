from job.passive import ERM
from environs import Env


def test_erm(wandb_cfg, dataset, train_cfg, test_cfg, model, mocker):
    mocker.patch("wandb.save")
    mocker.patch("wandb.log")
    mocker.patch("os.makedirs")
    mocker.patch("torch.save")

    erm = ERM(
        seed=Env().int("SEED"),
        dataset=dataset,
        wandb=wandb_cfg,
        train_cfg=train_cfg,
        test_cfg=test_cfg,
        model=model,
    )
    erm.run()
