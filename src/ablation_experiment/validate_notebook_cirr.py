from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data_utils import CIRRDataset
from src.utils import extract_index_features_with_text_captions, extract_index_features, \
    extract_index_features_with_text_captions_clip, extract_index_features_clip
from src.validate import generate_cirr_val_predictions
from src.validate_clip import generate_cirr_val_predictions as generate_cirr_val_predictions_clip


def compute_cirr_val_metrics_text_image_combinations(
    relative_val_dataset: CIRRDataset,
    blip_text_encoder: torch.nn.Module,
    multiple_text_index_features: List[torch.Tensor],
    multiple_text_index_names: List[List[str]],
    image_index_features: torch.Tensor,
    image_index_names: List[str],
    combining_function: callable,
    beta: float,
) -> pd.DataFrame:
    """
    Compute the CIRR metrics for different text-image combinations.

    Args:
        relative_val_dataset: The CIRR dataset.
        blip_text_encoder: The text encoder.
        multiple_text_index_features: The text features.
        multiple_text_index_names: The text names.
        image_index_features: The image features.
        image_index_names: The image names.
        combining_function: The function to combine the text and image features.
        beta: The beta value for the F-beta score.

    Returns:
        The computed validation metrics
    """
    all_text_similarities = []
    results = []
    reference_names = None
    target_names = None
    group_members = None

    # Compute similarities for individual text features
    for text_features, text_names in zip(multiple_text_index_features, multiple_text_index_names):
        # Generate text predictions and normalize features
        predicted_text_features, reference_names, target_names, group_members = generate_cirr_val_predictions(
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
        predicted_image_features, _, _, _ = generate_cirr_val_predictions(
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
    # Sort the results
    sorted_indices = torch.argsort(merged_similarities, dim=-1, descending=True).cpu()
    sorted_index_names = np.array(
        image_index_names if image_index_names else multiple_text_index_names[0]
    )[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(
            np.array(reference_names),
            len(image_index_names)
        ).reshape(len(target_names), -1)
    )

    sorted_index_names = sorted_index_names[reference_mask].reshape(
        sorted_index_names.shape[0],
        sorted_index_names.shape[1] - 1
    )

    # Compute the ground-truth labels with respect to the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(
            np.array(target_names),
            len(sorted_index_names[0])
        ).reshape(len(target_names), -1)
    )

    # Compute the subset predictions and ground-truth labels
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

    results.append({
        "beta": beta,
        "recall_at1": recall_at1,
        "recall_at5": recall_at5,
        "recall_at10": recall_at10,
        "recall_at50": recall_at50,
        "group_recall_at1": group_recall_at1,
        "group_recall_at2": group_recall_at2,
        "group_recall_at3": group_recall_at3,
    })

    return pd.DataFrame(results)


def compute_cirr_val_metrics_text_image_combinations_clip(
    relative_val_dataset: CIRRDataset,
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
    Compute the CIRR metrics for different text-image combinations.

    Args:
        relative_val_dataset: The CIRR dataset.
        clip_text_encoder: The text encoder.
        clip_tokenizer: The tokenizer.
        multiple_text_index_features: The text features.
        multiple_text_index_names: The text names.
        image_index_features: The image features.
        image_index_names: The image names.
        combining_function: The function to combine the text and image features.
        beta: The beta value for the F-beta score.

    Returns:
        The computed validation metrics
    """
    all_text_similarities = []
    results = []
    reference_names = None
    target_names = None
    group_members = None

    # Compute similarities for individual text features
    for text_features, text_names in zip(multiple_text_index_features, multiple_text_index_names):
        # Generate text predictions and normalize features
        predicted_text_features, reference_names, target_names, group_members = generate_cirr_val_predictions_clip(
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
        predicted_image_features, _, _, _ = generate_cirr_val_predictions_clip(
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
    # Sort the results
    sorted_indices = torch.argsort(merged_similarities, dim=-1, descending=True).cpu()
    sorted_index_names = np.array(
        image_index_names if image_index_names else multiple_text_index_names[0]
    )[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(
            np.array(reference_names),
            len(image_index_names)
        ).reshape(len(target_names), -1)
    )

    sorted_index_names = sorted_index_names[reference_mask].reshape(
        sorted_index_names.shape[0],
        sorted_index_names.shape[1] - 1
    )

    # Compute the ground-truth labels with respect to the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(
            np.array(target_names),
            len(sorted_index_names[0])
        ).reshape(len(target_names), -1)
    )

    # Compute the subset predictions and ground-truth labels
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

    results.append({
        "beta": beta,
        "recall_at1": recall_at1,
        "recall_at5": recall_at5,
        "recall_at10": recall_at10,
        "recall_at50": recall_at50,
        "group_recall_at1": group_recall_at1,
        "group_recall_at2": group_recall_at2,
        "group_recall_at3": group_recall_at3,
    })

    return pd.DataFrame(results)


def cirr_val_retrieval_text_image_combinations(
    combining_function: callable,
    blip_text_encoder: torch.nn.Module,
    blip_img_encoder: torch.nn.Module,
    text_captions: List[dict],
    preprocess: callable,
    beta: float,
    cache: dict,
) -> pd.DataFrame:
    """
    Evaluate the CIRR retrieval performance for different text-image combinations.

    Args:
        combining_function: The function to combine the text and image features.
        blip_text_encoder: The text encoder.
        blip_img_encoder: The image encoder.
        text_captions: The text captions.
        preprocess: The preprocessing function.
        beta: The beta value for the F-beta score.
        cache: The cache dictionary.

    Returns:
        The computed validation metrics.
    """
    blip_text_encoder = blip_text_encoder.float().eval()
    blip_img_encoder = blip_img_encoder.float().eval()

    if 'index_cache' not in cache:
        # Define the validation datasets and extract the index features
        classic_val_dataset = CIRRDataset(
            'val',
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

        cache['index_cache'] = {
            "multiple_index_features": multiple_index_features,
            "multiple_index_names": multiple_index_names,
            "image_index_features": image_index_features,
            "image_index_names": image_index_names,
        }
    else:
        multiple_index_features = cache['index_cache']['multiple_index_features']
        multiple_index_names = cache['index_cache']['multiple_index_names']
        image_index_features = cache['index_cache']['image_index_features']
        image_index_names = cache['index_cache']['image_index_names']

    relative_val_dataset = CIRRDataset(
        'val',
        'relative',
        preprocess,
        no_print_output=True,
    )

    # Define the combinations of features to evaluate
    feature_combinations = [
        ([multiple_index_features[0]], [multiple_index_names[0]]),  # Only first set
        ([multiple_index_features[1]], [multiple_index_names[1]]),  # Only second set
        ([multiple_index_features[2]], [multiple_index_names[2]]),  # Only third set
        ([multiple_index_features[0], multiple_index_features[1]], [multiple_index_names[0], multiple_index_names[1]]),
        # First and second set
        ([multiple_index_features[1], multiple_index_features[2]], [multiple_index_names[1], multiple_index_names[2]]),
        # Second and third set
        ([multiple_index_features[0], multiple_index_features[2]], [multiple_index_names[0], multiple_index_names[2]]),
        # First and third set
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
        result = compute_cirr_val_metrics_text_image_combinations(
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


def cirr_val_retrieval_text_image_combinations_clip(
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
    Evaluate the CIRR retrieval performance for different text-image combinations.

    Args:
        combining_function: The function to combine the text and image features.
        clip_text_encoder: The text encoder.
        clip_img_encoder: The image encoder.
        clip_tokenizer: The tokenizer.
        text_captions: The text captions.
        preprocess: The preprocessing function.
        beta: The beta value for the F-beta score.
        cache: The cache dictionary.

    Returns:
        The computed validation metrics.
    """
    clip_text_encoder = clip_text_encoder.float().eval()
    clip_img_encoder = clip_img_encoder.float().eval()

    if 'index_cache' not in cache:
        # Define the validation datasets and extract the index features
        classic_val_dataset = CIRRDataset(
            'val',
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

        cache['index_cache'] = {
            "multiple_index_features": multiple_index_features,
            "multiple_index_names": multiple_index_names,
            "image_index_features": image_index_features,
            "image_index_names": image_index_names
        }
    else:
        multiple_index_features = cache['index_cache']["multiple_index_features"]
        multiple_index_names = cache['index_cache']["multiple_index_names"]
        image_index_features = cache['index_cache']["image_index_features"]
        image_index_names = cache['index_cache']["image_index_names"]

    relative_val_dataset = CIRRDataset(
        'val',
        'relative',
        preprocess,
        no_print_output=True,
    )

    # Define the combinations of features to evaluate
    feature_combinations = [
        ([multiple_index_features[0]], [multiple_index_names[0]]),  # Only first set
        ([multiple_index_features[1]], [multiple_index_names[1]]),  # Only second set
        ([multiple_index_features[2]], [multiple_index_names[2]]),  # Only third set
        ([multiple_index_features[0], multiple_index_features[1]], [multiple_index_names[0], multiple_index_names[1]]),
        # First and second set
        ([multiple_index_features[1], multiple_index_features[2]], [multiple_index_names[1], multiple_index_names[2]]),
        # Second and third set
        ([multiple_index_features[0], multiple_index_features[2]], [multiple_index_names[0], multiple_index_names[2]]),
        # First and third set
        (multiple_index_features, multiple_index_names)  # All sets
    ]

    combination_name = [
        'First set',
        'Second set',
        'Third set',
        'First and second set',
        'Second and third set',
        'First and third set',
        'All sets',
    ]

    results = []

    for idx, (features_combination, names_combination) in tqdm(
        enumerate(feature_combinations),
        desc="Evaluating feature combinations",
        total=len(feature_combinations),
    ):
        result = compute_cirr_val_metrics_text_image_combinations_clip(
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