from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data_utils import FashionIQDataset
from src.utils import extract_index_features_with_text_captions_clip, extract_index_features_clip, \
    extract_index_features_with_text_captions, extract_index_features
from src.validate import generate_fiq_val_predictions
from src.validate_clip import generate_fiq_val_predictions as generate_fiq_val_predictions_clip


def compute_fiq_val_metrics_text_image_combinations(
    relative_val_dataset: FashionIQDataset,
    blip_text_encoder: torch.nn.Module,
    multiple_text_index_features: List[torch.Tensor],
    multiple_text_index_names: List[List[str]],
    image_index_features: torch.Tensor,
    image_index_names: List[str],
    combining_function: callable,
    beta: float,
) -> pd.DataFrame:
    """
    Compute validation metrics on FashionIQ dataset combining text and image similarities.

    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param blip_text_encoder: BLIP text encoder
    :param multiple_text_index_features: validation index features from text
    :param multiple_text_index_names: validation index names from text
    :param image_index_features: validation image index features
    :param image_index_names: validation image index names
    :param combining_function: function that combines features
    :param beta: beta value for the combination of text and image similarities
    :return: the computed validation metrics
    """
    all_text_similarities = []
    results = []
    target_names = None

    # Compute similarities for individual text features
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

        # Compute cosine similarity
        cosine_similarities = torch.mm(predicted_text_features, text_features.T)
        all_text_similarities.append(cosine_similarities)

    # Normalize and compute similarities for image features if available
    if image_index_features is not None and len(image_index_features) > 0:
        predicted_image_features, _ = generate_fiq_val_predictions(
            blip_text_encoder,
            relative_val_dataset,
            combining_function,
            image_index_names,
            image_index_features,
            no_print_output=True,
        )

        # Normalize and compute similarities
        image_index_features = F.normalize(image_index_features, dim=-1).float()
        image_similarities = predicted_image_features @ image_index_features.T
    else:
        image_similarities = torch.zeros_like(all_text_similarities[0])

    # Merge text similarities
    merged_text_similarities = torch.mean(torch.stack(all_text_similarities), dim=0)

    merged_similarities = beta * merged_text_similarities + (1 - beta) * image_similarities
    sorted_indices = torch.argsort(merged_similarities, dim=-1, descending=True).cpu()
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


def compute_fiq_val_metrics_text_image_combinations_clip(
    relative_val_dataset: FashionIQDataset,
    clip_text_encoder: torch.nn.Module,
    clip_tokenizer: callable,
    multiple_text_index_features: List[torch.Tensor],
    multiple_text_index_names: List[List[str]],
    image_index_features: torch.Tensor,
    image_index_names: List[str],
    combining_function: callable,
    beta: float,
) -> pd.DataFrame:
    """
    Compute validation metrics on FashionIQ dataset combining text and image similarities.

    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param clip_text_encoder: CLIP text encoder
    :param clip_tokenizer: CLIP tokenizer
    :param multiple_text_index_features: validation index features from text
    :param multiple_text_index_names: validation index names from text
    :param image_index_features: validation image index features
    :param image_index_names: validation image index names
    :param combining_function: function that combines features
    :param beta: beta value for the combination of text and image similarities
    :return: the computed validation metrics
    """
    all_text_similarities = []
    results = []
    target_names = None

    # Compute similarities for individual text features
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

        # Compute cosine similarity
        cosine_similarities = torch.mm(predicted_text_features, text_features.T)
        all_text_similarities.append(cosine_similarities)

    # Normalize and compute similarities for image features if available
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

        # Normalize and compute similarities
        image_index_features = F.normalize(image_index_features, dim=-1).float()
        image_similarities = predicted_image_features @ image_index_features.T
    else:
        image_similarities = torch.zeros_like(all_text_similarities[0])

    # Merge text similarities
    merged_text_similarities = torch.mean(torch.stack(all_text_similarities), dim=0)

    merged_similarities = beta * merged_text_similarities + (1 - beta) * image_similarities
    sorted_indices = torch.argsort(merged_similarities, dim=-1, descending=True).cpu()
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


def fiq_val_retrieval_text_image_combinations(
    dress_type: str,
    combining_function: callable,
    blip_text_encoder: torch.nn.Module,
    blip_img_encoder: torch.nn.Module,
    text_captions: List[dict],
    preprocess: callable,
    beta: float,
    cache: dict,
) -> pd.DataFrame:
    """
    Perform retrieval on FashionIQ validation set computing the metrics for different text feature combinations.

    :param dress_type: FashionIQ category on which perform the retrieval
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined features
    :param blip_text_encoder: BLIP text model
    :param blip_img_encoder: BLIP image model
    :param text_captions: text captions for the FashionIQ dataset
    :param preprocess: preprocess pipeline
    :param beta: beta value for the combination of text and image similarities
    :param cache: cache dictionary
    :return: DataFrame containing the retrieval metrics for each combination of text features
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

    # Define the combinations of features to evaluate
    feature_combinations = [
        ([multiple_index_features[0]], [multiple_index_names[0]]),  # Only first set
        ([multiple_index_features[1]], [multiple_index_names[1]]),  # Only second set
        ([multiple_index_features[2]], [multiple_index_names[2]]),  # Only third set
        ([multiple_index_features[0], multiple_index_features[1]], [multiple_index_names[0], multiple_index_names[1]]),  # First and second set
        ([multiple_index_features[1], multiple_index_features[2]], [multiple_index_names[1], multiple_index_names[2]]),  # Second and third set
        ([multiple_index_features[0], multiple_index_features[2]], [multiple_index_names[0], multiple_index_names[2]]),  # First and third set
        (multiple_index_features, multiple_index_names)  # All sets
    ]

    results = []

    combination_name = [
        'First set',
        'Second set',
        'Third set',
        'First and second set',
        'Second and third set',
        'First and third set',
        'All sets',
    ]

    for idx, (features_combination, names_combination) in tqdm(
        enumerate(feature_combinations),
        desc="Evaluating feature combinations",
        total=len(feature_combinations),
    ):
        result = compute_fiq_val_metrics_text_image_combinations(
            relative_val_dataset,
            blip_text_encoder,
            features_combination,
            names_combination,
            image_index_features,
            image_index_names,
            combining_function,
            beta,
        )
        result['Combination'] = combination_name[idx]
        results.append(result)

    # Concatenate all results into a single DataFrame
    return pd.concat(results, ignore_index=True)


def fiq_val_retrieval_text_image_combinations_clip(
    dress_type: str,
    combining_function: callable,
    clip_text_encoder: torch.nn.Module,
    clip_img_encoder: torch.nn.Module,
    clip_tokenizer: callable,
    text_captions: List[dict],
    preprocess: callable,
    beta: float,
    cache: dict,
) -> pd.DataFrame:
    """
    Perform retrieval on FashionIQ validation set computing the metrics for different text feature combinations.

    :param dress_type: FashionIQ category on which perform the retrieval
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined features
    :param clip_text_encoder: CLIP text model
    :param clip_img_encoder: CLIP image model
    :param clip_tokenizer: CLIP tokenizer
    :param text_captions: text captions for the FashionIQ dataset
    :param preprocess: preprocess pipeline
    :param beta: beta value for the combination of text and image similarities
    :param cache: cache dictionary
    :return: DataFrame containing the retrieval metrics for each combination of text features
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

    # Define the combinations of features to evaluate
    feature_combinations = [
        ([multiple_index_features[0]], [multiple_index_names[0]]),  # Only first set
        ([multiple_index_features[1]], [multiple_index_names[1]]),  # Only second set
        ([multiple_index_features[2]], [multiple_index_names[2]]),  # Only third set
        ([multiple_index_features[0], multiple_index_features[1]], [multiple_index_names[0], multiple_index_names[1]]),  # First and second set
        ([multiple_index_features[1], multiple_index_features[2]], [multiple_index_names[1], multiple_index_names[2]]),  # Second and third set
        ([multiple_index_features[0], multiple_index_features[2]], [multiple_index_names[0], multiple_index_names[2]]),  # First and third set
        (multiple_index_features, multiple_index_names)  # All sets
    ]

    results = []
    combination_name = [
        'First set',
        'Second set',
        'Third set',
        'First and second set',
        'Second and third set',
        'First and third set',
        'All sets',
    ]
    for idx, (features_combination, names_combination) in tqdm(
        enumerate(feature_combinations),
        desc="Evaluating feature combinations",
        total=len(feature_combinations),
    ):
        result = compute_fiq_val_metrics_text_image_combinations_clip(
            relative_val_dataset,
            clip_text_encoder,
            clip_tokenizer,
            features_combination,
            names_combination,
            image_index_features,
            image_index_names,
            combining_function,
            beta,
        )
        result['Combination'] = combination_name[idx]
        results.append(result)

    # Concatenate all results into a single DataFrame
    return pd.concat(results, ignore_index=True)
