from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data_utils import FashionIQDataset
from src.utils import extract_index_features_with_text_captions, extract_index_features, \
    extract_index_features_with_text_captions_clip, extract_index_features_clip
from src.validate import generate_fiq_val_predictions
from src.validate_clip import generate_fiq_val_predictions as generate_fiq_val_predictions_clip


def compute_fiq_val_metrics_text_image_grid_search(
    relative_val_dataset: FashionIQDataset,
    blip_text_encoder: torch.nn.Module,
    multiple_text_index_features: List[torch.Tensor],
    multiple_text_index_names: List[List[str]],
    image_index_features: torch.Tensor,
    image_index_names: List[str],
    combining_function: callable,
) -> pd.DataFrame:
    """
    Compute validation metrics on FashionIQ dataset combining text and image distances.
    The only difference with the original function is that we grid search over beta values in the range [0, 1].
    And the return a store dataframe with the results.

    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param blip_text_encoder: BLIP model
    :param multiple_text_index_features: validation index features from text
    :param multiple_text_index_names: validation index names from text
    :param image_index_features: validation image index features
    :param image_index_names: validation image index names
    :param combining_function: function that combines features
    :return: the computed validation metrics
    """
    all_text_distances = []
    results = []
    betas = np.arange(0, 1.05, 0.05)
    target_names = None

    # Compute distances for individual text features
    for text_features, text_names in zip(multiple_text_index_features, multiple_text_index_names):
        # Generate text predictions and normalize features
        predicted_text_features, target_names = generate_fiq_val_predictions(
            blip_text_encoder,
            relative_val_dataset,
            combining_function,
            text_names,
            text_features,
            no_print_output=True,
        )
        # Normalize features
        text_features = F.normalize(text_features, dim=-1)
        predicted_text_features = F.normalize(predicted_text_features, dim=-1)

        # Compute cosine similarity and convert to distance
        cosine_similarities = torch.mm(predicted_text_features, text_features.T)
        distances = 1 - cosine_similarities
        all_text_distances.append(distances)

    # Normalize and compute distances for image features if available
    if image_index_features is not None and len(image_index_features) > 0:
        predicted_image_features, _ = generate_fiq_val_predictions(
            blip_text_encoder,
            relative_val_dataset,
            combining_function,
            image_index_names,
            image_index_features,
            no_print_output=True,
        )

        # Normalize and compute distances
        image_index_features = F.normalize(image_index_features, dim=-1).float()
        image_distances = 1 - predicted_image_features @ image_index_features.T
    else:
        image_distances = torch.zeros_like(all_text_distances[0])

    # Merge text distances
    merged_text_distances = torch.mean(torch.stack(all_text_distances), dim=0)

    # Iterating over beta values
    for beta in betas:
        merged_distances = beta * merged_text_distances + (1 - beta) * image_distances
        sorted_indices = torch.argsort(merged_distances, dim=-1).cpu()
        sorted_index_names = np.array(image_index_names if image_index_names else multiple_text_index_names[0])[
            sorted_indices]
        labels = torch.tensor(
            sorted_index_names == np.repeat(
                np.array(target_names),
                len(image_index_names if image_index_names else multiple_text_index_names[0])
            ).reshape(len(target_names), -1)
        )
        assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
        recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
        recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
        results.append({"beta": beta, "recall_at10": recall_at10, "recall_at50": recall_at50})

    return pd.DataFrame(results)

def compute_fiq_val_metrics_text_image_grid_search_clip(
    relative_val_dataset: FashionIQDataset,
    clip_text_encoder: torch.nn.Module,
    clip_tokenizer: callable,
    multiple_text_index_features: List[torch.Tensor],
    multiple_text_index_names: List[List[str]],
    image_index_features: torch.Tensor,
    image_index_names: List[str],
    combining_function: callable,
) -> pd.DataFrame:
    """
    Compute validation metrics on FashionIQ dataset combining text and image distances.

    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param clip_text_encoder: CLIP text encoder
    :param clip_tokenizer: CLIP tokenizer
    :param multiple_text_index_features: validation index features from text
    :param multiple_text_index_names: validation index names from text
    :param image_index_features: validation image index features
    :param image_index_names: validation image index names
    :param combining_function: function that combines features
    :return: the computed validation metrics
    """
    all_text_distances = []
    results = []
    betas = np.arange(0, 1.05, 0.05)
    target_names = None

    # Compute distances for individual text features
    for text_features, text_names in zip(multiple_text_index_features, multiple_text_index_names):
        # Generate text predictions and normalize features
        predicted_text_features, target_names = generate_fiq_val_predictions_clip(
            clip_text_encoder,
            clip_tokenizer,
            relative_val_dataset,
            combining_function,
            text_names,
            text_features,
            no_print_output=True,
        )
        # Normalize features
        text_features = F.normalize(text_features, dim=-1)
        predicted_text_features = F.normalize(predicted_text_features, dim=-1)

        # Compute cosine similarity and convert to distance
        cosine_similarities = torch.mm(predicted_text_features, text_features.T)
        distances = 1 - cosine_similarities
        all_text_distances.append(distances)

    # Normalize and compute distances for image features if available
    if image_index_features is not None and len(image_index_features) > 0:
        predicted_image_features, _ = generate_fiq_val_predictions_clip(
            clip_text_encoder,
            clip_tokenizer,
            relative_val_dataset,
            combining_function,
            image_index_names,
            image_index_features,
            no_print_output=True,
        )

        # Normalize and compute distances
        image_index_features = F.normalize(image_index_features, dim=-1).float()
        image_distances = 1 - predicted_image_features @ image_index_features.T
    else:
        image_distances = torch.zeros_like(all_text_distances[0])

    # Merge text distances
    merged_text_distances = torch.mean(torch.stack(all_text_distances), dim=0)

    # Iterating over beta values
    for beta in betas:
        merged_distances = beta * merged_text_distances + (1 - beta) * image_distances
        sorted_indices = torch.argsort(merged_distances, dim=-1).cpu()
        sorted_index_names = np.array(
            image_index_names if image_index_names else multiple_text_index_names[0]
        )[sorted_indices]

        labels = torch.tensor(
            sorted_index_names == np.repeat(
                np.array(target_names),
                len(image_index_names if image_index_names else multiple_text_index_names[0])
            ).reshape(len(target_names), -1)
        )
        assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
        recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
        recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
        results.append({"beta": beta, "recall_at10": recall_at10, "recall_at50": recall_at50})

    return pd.DataFrame(results)


def fiq_val_retrieval_text_image_grid_search(
    dress_type: str,
    combining_function: callable,
    blip_text_encoder: torch.nn.Module,
    blip_img_encoder: torch.nn.Module,
    text_captions: List[dict],
    preprocess: callable,
    cache: dict,
) -> pd.DataFrame:
    """
    Perform retrieval on FashionIQ validation set computing the metrics. To combine the features, the `combining_function` is used
    :param dress_type: FashionIQ category on which perform the retrieval
    :param combining_function:function which takes as input (image_features, text_features) and outputs the combined features
    :param blip_text_encoder: BLIP text model
    :param blip_img_encoder: BLIP image model
    :param text_captions: text captions for the FashionIQ dataset
    :param preprocess: preprocess pipeline
    :param cache: cache dictionary
    """
    cache_key = f"{dress_type}_cache"

    blip_text_encoder = blip_text_encoder.float().eval()
    blip_img_encoder = blip_img_encoder.float().eval()

    if cache_key not in cache:
        # Define the validation datasets and extract the index features
        classic_val_dataset = FashionIQDataset(
            'val',
            [dress_type],
            'classic',
            preprocess,
            no_print_output=True,
        )

        multiple_index_features, multiple_index_names = [], []

        for i in range(3):
            index_features, index_names, _ = extract_index_features_with_text_captions(
                classic_val_dataset,
                blip_text_encoder,
                text_captions,
                i + 1,
                no_print_output=True,
            )
            multiple_index_features.append(index_features)
            multiple_index_names.append(index_names)

        image_index_features, image_index_names = extract_index_features(
            classic_val_dataset,
            blip_img_encoder,
            no_print_output=True,
        )

        cache[cache_key] = {
            "multiple_index_features": multiple_index_features,
            "multiple_index_names": multiple_index_names,
            "image_index_features": image_index_features,
            "image_index_names": image_index_names
        }
    else:
        multiple_index_features = cache[cache_key]["multiple_index_features"]
        multiple_index_names = cache[cache_key]["multiple_index_names"]
        image_index_features = cache[cache_key]["image_index_features"]
        image_index_names = cache[cache_key]["image_index_names"]

    relative_val_dataset = FashionIQDataset(
        'val',
        [dress_type],
        'relative',
        preprocess,
        no_print_output=True,
    )

    return compute_fiq_val_metrics_text_image_grid_search(
        relative_val_dataset,
        blip_text_encoder,
        multiple_index_features,
        multiple_index_names,
        image_index_features,
        image_index_names,
        combining_function
    )


def fiq_val_retrieval_text_image_grid_search_clip(
    dress_type: str,
    combining_function: callable,
    clip_text_encoder: torch.nn.Module,
    clip_img_encoder: torch.nn.Module,
    clip_tokenizer: callable,
    text_captions: List[dict],
    preprocess: callable,
    cache: dict,
) -> pd.DataFrame:
    """
    Perform retrieval on FashionIQ validation set computing the metrics. To combine the features, the `combining_function` is used
    :param dress_type: FashionIQ category on which perform the retrieval
    :param combining_function:function which takes as input (image_features, text_features) and outputs the combined features
    :param clip_text_encoder: CLIP text model
    :param clip_img_encoder: CLIP image model
    :param clip_tokenizer: CLIP tokenizer
    :param text_captions: text captions for the FashionIQ dataset
    :param preprocess: preprocess pipeline
    :param cache: cache dictionary
    """
    cache_key = f"{dress_type}_cache"

    clip_text_encoder = clip_text_encoder.float().eval()
    clip_img_encoder = clip_img_encoder.float().eval()

    if cache_key not in cache:
        # Define the validation datasets and extract the index features
        classic_val_dataset = FashionIQDataset(
            'val',
            [dress_type],
            'classic',
            preprocess,
            no_print_output=True,
        )

        multiple_index_features, multiple_index_names = [], []

        for i in range(3):
            index_features, index_names, _ = extract_index_features_with_text_captions_clip(
                classic_val_dataset,
                clip_text_encoder,
                clip_tokenizer,
                text_captions,
                i + 1,
                no_print_output=True,
            )
            multiple_index_features.append(index_features)
            multiple_index_names.append(index_names)

        image_index_features, image_index_names = extract_index_features_clip(
            classic_val_dataset,
            clip_img_encoder,
            no_print_output=True,
        )

        cache[cache_key] = {
            "multiple_index_features": multiple_index_features,
            "multiple_index_names": multiple_index_names,
            "image_index_features": image_index_features,
            "image_index_names": image_index_names
        }
    else:
        multiple_index_features = cache[cache_key]["multiple_index_features"]
        multiple_index_names = cache[cache_key]["multiple_index_names"]
        image_index_features = cache[cache_key]["image_index_features"]
        image_index_names = cache[cache_key]["image_index_names"]

    relative_val_dataset = FashionIQDataset(
        'val',
        [dress_type],
        'relative',
        preprocess,
        no_print_output=True,
    )

    return compute_fiq_val_metrics_text_image_grid_search_clip(
        relative_val_dataset,
        clip_text_encoder,
        clip_tokenizer,
        multiple_index_features,
        multiple_index_names,
        image_index_features,
        image_index_names,
        combining_function,
    )


def compute_results_fiq_val(
    dress_type: str,
    combining_function: callable,
    blip_text_encoder: torch.nn.Module,
    blip_img_encoder: torch.nn.Module,
    text_captions: List[dict],
    preprocess: callable,
    beta: float = 0.65,
    cache: dict = None,
) -> Tuple[torch.Tensor, List[str], List[str]]:
    """
    Compute results for the FashionIQ dataset. This is used to create a plot to visualize the results.
    And use to compare the results with the original implementation.

    :param dress_type: FashionIQ category on which performs the retrieval
    :param combining_function:function which takes as input (image_features, text_features) and outputs the combined features
    :param blip_text_encoder: BLIP text model
    :param blip_img_encoder: BLIP image model
    :param text_captions: text captions for the FashionIQ dataset
    :param preprocess: preprocess pipeline
    :param beta: beta value for combining text and image distances
    :param cache: cache dictionary

    :return: [(image_path, text_caption, candidate_image_paths)]
    """
    if cache is None:
        cache = {}
    cache_key = f"{dress_type}_cache"
    blip_text_encoder = blip_text_encoder.float().eval()
    blip_img_encoder = blip_img_encoder.float().eval()

    if cache_key not in cache:
        # Define the validation datasets and extract the index features
        classic_val_dataset = FashionIQDataset(
            'val',
            [dress_type],
            'classic',
            preprocess,
            no_print_output=True,
        )

        multiple_index_features, multiple_index_names = [], []

        for i in range(3):
            index_features, index_names, _ = extract_index_features_with_text_captions(
                classic_val_dataset,
                blip_text_encoder,
                text_captions,
                i + 1,
                no_print_output=True,
            )
            multiple_index_features.append(index_features)
            multiple_index_names.append(index_names)

        image_index_features, image_index_names = extract_index_features(
            classic_val_dataset,
            blip_img_encoder,
            no_print_output=True,
        )

        cache[cache_key] = {
            "multiple_index_features": multiple_index_features,
            "multiple_index_names": multiple_index_names,
            "image_index_features": image_index_features,
            "image_index_names": image_index_names
        }
    else:
        multiple_index_features = cache[cache_key]["multiple_index_features"]
        multiple_index_names = cache[cache_key]["multiple_index_names"]
        image_index_features = cache[cache_key]["image_index_features"]
        image_index_names = cache[cache_key]["image_index_names"]

    relative_val_dataset = FashionIQDataset(
        'val',
        [dress_type],
        'relative',
        preprocess,
        no_print_output=True,
    )

    all_text_distances = []

    # Compute distances for individual text features
    for text_features, text_names in zip(multiple_index_features, multiple_index_names):
        # Generate text predictions and normalize features
        predicted_text_features, target_names = generate_fiq_val_predictions(
            blip_text_encoder,
            relative_val_dataset,
            combining_function,
            text_names,
            text_features,
            no_print_output=True,
        )
        # Normalize features
        text_features = F.normalize(text_features, dim=-1)
        predicted_text_features = F.normalize(predicted_text_features, dim=-1)

        # Compute cosine similarity and convert to distance
        cosine_similarities = torch.mm(predicted_text_features, text_features.T)
        distances = 1 - cosine_similarities
        all_text_distances.append(distances)

    predicted_image_features, target_names = generate_fiq_val_predictions(
        blip_text_encoder,
        relative_val_dataset,
        combining_function,
        image_index_names,
        image_index_features,
        no_print_output=True,
    )

    # Normalize and compute distances
    image_index_features = F.normalize(image_index_features, dim=-1).float()
    image_distances = 1 - predicted_image_features @ image_index_features.T

    # Merge text distances
    merged_text_distances = torch.mean(torch.stack(all_text_distances), dim=0)

    merged_distances = beta * merged_text_distances + (1 - beta) * image_distances

    sorted_indices = torch.argsort(merged_distances, dim=-1).cpu()
    sorted_index_names = np.array(image_index_names if image_index_names else multiple_index_names[0])[sorted_indices]
    labels = torch.tensor(
        sorted_index_names == np.repeat(
            np.array(target_names),
            len(image_index_names if image_index_names else multiple_index_names[0])
        ).reshape(len(target_names), -1)
    )
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    print(f"Recall at 10: {recall_at10}")
    print(f"Recall at 50: {recall_at50}")

    return sorted_indices, image_index_names, target_names


def compute_results_fiq_val_clip(
    dress_type: str,
    combining_function: callable,
    clip_text_encoder: torch.nn.Module,
    clip_img_encoder: torch.nn.Module,
    clip_tokenizer: callable,
    text_captions: List[dict],
    preprocess: callable,
    beta: float = 0.65,
    cache: dict = None,
) -> Tuple[torch.tensor, List[str], List[str]]:
    """
    View sample results on FashionIQ dataset combining text and image distances.

    :param dress_type: FashionIQ category on which performs the retrieval
    :param combining_function: Function which takes as input (image_features, text_features)
                               and outputs the combined features
    :param clip_text_encoder: CLIP text model
    :param clip_img_encoder: CLIP image model
    :param clip_tokenizer: CLIP tokenizer
    :param text_captions: text captions for the FashionIQ dataset
    :param preprocess: preprocess pipeline
    :param beta: beta value for combining text and image distances
    :param cache: cache dictionary

    :return: [(image_path, text_caption, candidate_image_paths)]
    """
    cache_key = f"{dress_type}_cache"
    clip_text_encoder = clip_text_encoder.float().eval()
    clip_img_encoder = clip_img_encoder.float().eval()

    if cache_key not in cache:
        # Define the validation datasets and extract the index features
        classic_val_dataset = FashionIQDataset(
            'val',
            [dress_type],
            'classic',
            preprocess,
            no_print_output=True,
        )

        multiple_index_features, multiple_index_names = [], []

        for i in range(3):
            index_features, index_names, _ = extract_index_features_with_text_captions_clip(
                classic_val_dataset,
                clip_text_encoder,
                clip_tokenizer,
                text_captions,
                i + 1,
                no_print_output=True,
            )
            multiple_index_features.append(index_features)
            multiple_index_names.append(index_names)

        image_index_features, image_index_names = extract_index_features_clip(
            classic_val_dataset,
            clip_img_encoder,
            no_print_output=True,
        )
    else:
        multiple_index_features = cache[cache_key]["multiple_index_features"]
        multiple_index_names = cache[cache_key]["multiple_index_names"]
        image_index_features = cache[cache_key]["image_index_features"]
        image_index_names = cache[cache_key]["image_index_names"]

    relative_val_dataset = FashionIQDataset(
        'val',
        [dress_type],
        'relative',
        preprocess,
        no_print_output=True,
    )

    all_text_distances = []

    # Compute distances for individual text features
    for text_features, text_names in zip(multiple_index_features, multiple_index_names):
        # Generate text predictions and normalize features
        predicted_text_features, target_names = generate_fiq_val_predictions_clip(
            clip_text_encoder,
            clip_tokenizer,
            relative_val_dataset,
            combining_function,
            text_names,
            text_features,
            no_print_output=True,
        )
        # Normalize features
        text_features = F.normalize(text_features, dim=-1)
        predicted_text_features = F.normalize(predicted_text_features, dim=-1)

        # Compute cosine similarity and convert to distance
        cosine_similarities = torch.mm(predicted_text_features, text_features.T)
        distances = 1 - cosine_similarities
        all_text_distances.append(distances)

    predicted_image_features, target_names = generate_fiq_val_predictions_clip(
        clip_text_encoder,
        clip_tokenizer,
        relative_val_dataset,
        combining_function,
        image_index_names,
        image_index_features,
        no_print_output=True,
    )

    # Normalize and compute distances
    image_index_features = F.normalize(image_index_features, dim=-1).float()
    image_distances = 1 - predicted_image_features @ image_index_features.T

    # Merge text distances
    merged_text_distances = torch.mean(torch.stack(all_text_distances), dim=0)

    merged_distances = beta * merged_text_distances + (1 - beta) * image_distances

    sorted_indices = torch.argsort(merged_distances, dim=-1).cpu()
    sorted_index_names = np.array(image_index_names if image_index_names else multiple_index_names[0])[sorted_indices]
    labels = torch.tensor(
        sorted_index_names == np.repeat(
            np.array(target_names),
            len(image_index_names if image_index_names else multiple_index_names[0])
        ).reshape(len(target_names), -1)
    )
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    print(f"Recall at 10: {recall_at10}")
    print(f"Recall at 50: {recall_at50}")

    return sorted_indices, image_index_names, target_names