from itertools import repeat

from hydra.utils import instantiate

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed


def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def get_evaluation_batches(index, batch_size, single_item_above_seconds):
    batches = []
    batch = []
    for idx, record in enumerate(index):
        if record["audio_len"] > single_item_above_seconds:
            if batch:
                batches.append(batch)
                batch = []
            batches.append([idx])
        else:
            batch.append(idx)
            if len(batch) == batch_size:
                batches.append(batch)
                batch = []
    if batch:
        batches.append(batch)
    return batches


def get_dataloaders(config, text_encoder, device):
    """
    Create dataloaders for each of the dataset partitions.
    Also creates instance and batch transforms.

    Args:
        config (DictConfig): hydra experiment config.
        text_encoder (CTCTextEncoder): instance of the text encoder
            for the datasets.
        device (str): device to use for batch transforms.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for a
            partition defined by key.
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
    """
    # transforms or augmentations init
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    # dataloaders init
    dataloaders = {}
    for dataset_partition in config.datasets.keys():
        # dataset partition init
        dataset = instantiate(
            config.datasets[dataset_partition], text_encoder=text_encoder
        )  # instance transforms are defined inside
        if len(dataset) == 0:
            raise ValueError(f"Dataset partition {dataset_partition} contains no audio")

        batch_size = config.dataloader.batch_size
        if dataset_partition != "train":
            batch_size = min(
                config.get("evaluation_batch_size", batch_size), len(dataset)
            )

        assert batch_size <= len(dataset), (
            f"The batch size ({batch_size}) cannot "
            f"be larger than the dataset length ({len(dataset)})"
        )

        loader_kwargs = dict(
            batch_size=batch_size,
            drop_last=(dataset_partition == "train"),
            shuffle=(dataset_partition == "train"),
        )
        threshold = config.get("evaluation_single_item_above_seconds")
        if dataset_partition != "train" and threshold is not None:
            loader_kwargs["batch_sampler"] = get_evaluation_batches(
                dataset._index, batch_size, threshold
            )
            loader_kwargs["batch_size"] = 1  # required with a custom batch_sampler

        partition_dataloader = instantiate(
            config.dataloader,
            dataset=dataset,
            collate_fn=collate_fn,
            worker_init_fn=set_worker_seed,
            **loader_kwargs,
        )
        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders, batch_transforms
