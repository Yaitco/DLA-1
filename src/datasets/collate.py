import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    B = len(dataset_items)
    F = dataset_items[0]["spectrogram"].shape[-2]

    audio_length = torch.tensor([item["audio"].shape[-1] for item in dataset_items])
    audio_batch = torch.zeros((B, 1, audio_length.max()))

    spectrogram_length = torch.tensor(
        [item["spectrogram"].shape[-1] for item in dataset_items]
    )
    spectrogram_batch = torch.zeros((B, F, max(spectrogram_length)))

    text_encoded_length = torch.tensor(
        [item["text_encoded"].shape[-1] for item in dataset_items]
    )
    text_encoded_batch = torch.zeros((B, max(text_encoded_length)), dtype=torch.long)

    text_batch = [item["text"] for item in dataset_items]
    audio_path_batch = [item["audio_path"] for item in dataset_items]

    for i, item in enumerate(dataset_items):
        audio, spectrogram, text_encoded = (
            item["audio"],
            item["spectrogram"],
            item["text_encoded"],
        )

        audio_batch[i, :, : audio_length[i]] = audio
        spectrogram_batch[i, :, : spectrogram_length[i]] = spectrogram.squeeze(0)
        text_encoded_batch[i, : text_encoded_length[i]] = text_encoded.squeeze(0)

    return {
        "audio": audio_batch,
        "audio_length": audio_length,
        "spectrogram": spectrogram_batch,
        "spectrogram_length": spectrogram_length,
        "text_encoded": text_encoded_batch,
        "text_encoded_length": text_encoded_length,
        "text": text_batch,
        "audio_path": audio_path_batch,
    }
