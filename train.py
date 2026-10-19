import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup text_encoder
    text_encoder = instantiate(config.text_encoder)

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, text_encoder, device)

    # build model architecture, then print to console
    model = instantiate(config.model, n_tokens=len(text_encoder))
    encoder_checkpoint = config.trainer.get("encoder_checkpoint")
    if (
        encoder_checkpoint is not None
        and config.trainer.get("resume_from") is None
        and config.trainer.get("from_pretrained") is None
    ):
        model.load_encoder_from_ctc(encoder_checkpoint)
        logger.info("Loaded CTC encoder from %s", encoder_checkpoint)
    model = model.to(device)
    logger.info(model)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)

    metrics = {"train": [], "inference": []}
    for metric_type in ["train", "inference"]:
        for metric_config in config.metrics.get(metric_type, []):
            # use text_encoder in metrics
            metrics[metric_type].append(
                instantiate(metric_config, text_encoder=text_encoder)
            )

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    encoder_lr = config.trainer.get("encoder_lr")
    if encoder_lr is not None:
        trainable_params = [
            {
                "params": [p for p in model.encoder.parameters() if p.requires_grad],
                "lr": encoder_lr,
            },
            {
                "params": [
                    p
                    for name, p in model.named_parameters()
                    if p.requires_grad and not name.startswith("encoder.")
                ],
            },
        ]
    optimizer = instantiate(config.optimizer, params=trainable_params, _convert_="all")
    lr_scheduler = instantiate(
        config.lr_scheduler, optimizer=optimizer, _convert_="all"
    )

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        text_encoder=text_encoder,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
