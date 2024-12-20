"""
 * Copyright (c) 2024
 * All rights reserved.
 * SPDX-License-Identifier: MIT
 * This file incorporates code from the Bi-Blip4CIR project, available at:
 * https://github.com/Cuberick-Orion/Bi-Blip4CIR
 * For full license text of the original code, see the LICENSE file in the root of the Bi-Blip4CIR repo.
"""
import json
import warnings
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from statistics import mean
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from rich import print

from src.data_utils import squarepad_transform, FashionIQDataset, targetpad_transform, CIRRDataset
from src.combiner import Combiner
from src.utils import extract_index_features, collate_fn, element_wise_sum, device, \
    extract_index_features_with_text_captions, element_wise_sum_with_alpha


def compute_fiq_val_metrics(
    relative_val_dataset: FashionIQDataset,
    blip_text_encoder: torch.nn.Module,
    index_features: torch.Tensor,
    index_names: List[str],
    combining_function: callable
) -> Tuple[float, float]:
    """
    Compute validation metrics on FashionIQ dataset
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param blip_text_encoder: BLIP model
    :param index_features: validation index features
    :param index_names: validation index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined features
    :return: the computed validation metrics
    """

    # Generate predictions
    predicted_features, target_names = generate_fiq_val_predictions(
        blip_text_encoder,
        relative_val_dataset,
        combining_function,
        index_names,
        index_features
    )

    print(f"[{datetime.now()}] Compute FashionIQ {relative_val_dataset.dress_types} validation metrics")

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the similarities and sort the results
    similarities = predicted_features @ index_features.T
    sorted_indices = torch.argsort(similarities, dim=-1, descending=True).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(
            np.array(target_names),
            len(index_names)
        ).reshape(len(target_names), -1)
    )
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    return recall_at10, recall_at50


def compute_fiq_val_metrics_text_image(
    relative_val_dataset: FashionIQDataset,
    blip_text_encoder: torch.nn.Module,
    multiple_text_index_features: List[torch.Tensor],
    multiple_text_index_names: List[List[str]],
    image_index_features: torch.Tensor,
    image_index_names: List[str],
    combining_function: callable,
    beta: float
) -> Tuple[float, float]:
    """
    Compute validation metrics on FashionIQ dataset combining text and image similarities.

    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param blip_text_encoder: BLIP model
    :param multiple_text_index_features: validation index features from text
    :param multiple_text_index_names: validation index names from text
    :param image_index_features: validation image index features
    :param image_index_names: validation image index names
    :param combining_function: function that combines features
    :param beta: weight for combining text and image similarities
    :return: the computed validation metrics
    """
    all_text_similarities = []
    target_names = None

    # Compute similarities for individual text features
    for text_features, text_names in zip(multiple_text_index_features, multiple_text_index_names):
        # Generate text predictions and normalize features
        predicted_text_features, target_names = generate_fiq_val_predictions(
            blip_text_encoder,
            relative_val_dataset,
            combining_function,
            text_names,
            text_features
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
            image_index_features
        )

        # Normalize and compute similarities
        image_index_features = F.normalize(image_index_features, dim=-1).float()
        image_similarities = predicted_image_features @ image_index_features.T
    else:
        image_similarities = torch.zeros_like(all_text_similarities[0])

    # Merge text similarities
    merged_text_similarities = torch.mean(torch.stack(all_text_similarities), dim=0)

    # Merge text and image similarities
    merged_similarities = beta * merged_text_similarities + (1 - beta) * image_similarities

    # Sort the results
    sorted_indices = torch.argsort(merged_similarities, dim=-1, descending=True).cpu()
    sorted_index_names = np.array(
        image_index_names if image_index_names else multiple_text_index_names[0]
    )[sorted_indices]

    # Compute the ground-truth labels with respect to the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(
            np.array(target_names),
            len(sorted_index_names[0])
        ).reshape(len(target_names), -1)
    )

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    return recall_at10, recall_at50


def generate_fiq_val_predictions(
    blip_text_encoder: torch.nn.Module,
    relative_val_dataset: FashionIQDataset,
    combining_function: callable,
    index_names: List[str],
    index_features: torch.Tensor,
    no_print_output: bool = False,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Compute FashionIQ predictions on the validation set
    :param blip_text_encoder: BLIP model
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined features
    :param index_names: validation index names
    :param index_features: validation index features
    :param no_print_output: whether to print the output
    :return: predicted features and target names
    """
    relative_val_loader = DataLoader(
        dataset=relative_val_dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=False,
        collate_fn=collate_fn,
        shuffle=False
    )

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize predicted features and target names
    predicted_features = torch.empty((0, 256)).to(device, non_blocking=True)
    target_names = []

    if not no_print_output:
        relative_val_loader = tqdm(relative_val_loader)

    for reference_names, batch_target_names, captions in relative_val_loader:  # Load data

        # Concatenate the captions in a deterministic way
        flattened_captions: list = np.array(captions).T.flatten().tolist()
        input_captions = [
            f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}" for
            i in range(0, len(flattened_captions), 2)]

        # Compute the predicted features
        with torch.no_grad():
            text_features = blip_text_encoder(input_captions, 77, device)
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if text_features.shape[0] == 1:
                reference_image_features = itemgetter(*reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*reference_names)(
                    name_to_feat))
                # To avoid unnecessary computation,
                # retrieve the reference image features directly from the index features
            batch_predicted_features = combining_function(reference_image_features, text_features)

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        target_names.extend(batch_target_names)

    return predicted_features, target_names


def fashioniq_val_retrieval(
    dress_type: str,
    combining_function: callable,
    blip_text_encoder: torch.nn.Module,
    blip_img_encoder: torch.nn.Module,
    preprocess: callable
) -> Tuple[float, float]:
    """
    Perform retrieval on FashionIQ validation set computing the metrics. To combine the features, the `combining_function` is used
    :param dress_type: FashionIQ category on which perform the retrieval
    :param combining_function:function which takes as input (image_features, text_features) and outputs the combined features
    :param blip_text_encoder: BLIP text model
    :param blip_img_encoder: BLIP image model
    :param preprocess: The preprocess pipeline
    """
    blip_text_encoder = blip_text_encoder.float().eval()
    blip_img_encoder = blip_img_encoder.float().eval()

    # Define the validation datasets and extract the index features
    classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess)
    index_features, index_names = extract_index_features(classic_val_dataset, blip_img_encoder)
    relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess)

    return compute_fiq_val_metrics(
        relative_val_dataset,
        blip_text_encoder,
        index_features,
        index_names,
        combining_function
    )


def fashioniq_val_retrieval_text_image(
    dress_type: str,
    combining_function: callable,
    blip_text_encoder: torch.nn.Module,
    blip_img_encoder: torch.nn.Module,
    text_captions: List[dict],
    beta: float,
    preprocess: callable):
    """
    Perform retrieval on FashionIQ validation set computing the metrics. To combine the features, the `combining_function` is used
    :param dress_type: FashionIQ category on which perform the retrieval
    :param combining_function:function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param blip_text_encoder: BLIP text model
    :param blip_img_encoder: BLIP image model
    :param text_captions: text captions for the FashionIQ dataset
    :param beta: weight for combining text and image similarities
    :param preprocess: preprocess pipeline
    """
    blip_text_encoder = blip_text_encoder.float().eval()
    blip_img_encoder = blip_img_encoder.float().eval()

    # Define the validation datasets and extract the index features
    classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess)

    multiple_index_features, multiple_index_names = [], []

    for i in range(3):
        index_features, index_names, _ = extract_index_features_with_text_captions(
            classic_val_dataset,
            blip_text_encoder,
            text_captions,
            i + 1
        )
        multiple_index_features.append(index_features)
        multiple_index_names.append(index_names)

    image_index_features, image_index_names = extract_index_features(classic_val_dataset, blip_img_encoder)

    relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess)

    return compute_fiq_val_metrics_text_image(
        relative_val_dataset,
        blip_text_encoder,
        multiple_index_features,
        multiple_index_names,
        image_index_features,
        image_index_names,
        combining_function,
        beta
    )


def compute_cirr_val_metrics(
    relative_val_dataset: CIRRDataset,
    blip_model: torch.nn.Module,
    index_features: torch.Tensor,
    index_names: List[str],
    combining_function: callable
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Compute validation metrics on CIRR dataset
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param blip_model: BLIP model
    :param index_features: validation index features
    :param index_names: validation index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined features
    :return: the computed validation metrics
    """
    # Generate predictions
    predicted_features, reference_names, target_names, group_members = generate_cirr_val_predictions(
        blip_model,
        relative_val_dataset,
        combining_function,
        index_names,
        index_features
    )

    print(f"[{datetime.now()}] Compute CIRR validation metrics")

    # Normalize the index features
    print(f"[{datetime.now()}] Compute the index features")
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the similarities and sort the results
    print(f"[{datetime.now()}] Compute the similarities and sort the results")
    similarities = predicted_features @ index_features.T
    sorted_indices = torch.argsort(similarities, dim=-1, descending=True).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(
            np.array(reference_names),
            len(index_names)
        ).reshape(len(target_names), -1)
    )
    sorted_index_names = sorted_index_names[reference_mask].reshape(
        sorted_index_names.shape[0],
        sorted_index_names.shape[1] - 1
    )
    # Compute the ground-truth labels wrt the predictions
    print(f"[{datetime.now()}] Compute the ground-truth labels wrt the predictions")
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1))

    # Compute the subset predictions and ground-truth labels
    print(f"[{datetime.now()}] Compute subset predictions and ground-truth labels")
    group_members = np.array(group_members)
    print(f"[{datetime.now()}] Compute group_mask")
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    print(f"[{datetime.now()}] Compute group_labels")
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    print(f"[{datetime.now()}] Compute assert torch.equal")
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    print(f"[{datetime.now()}] Compute metrics")
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

    return group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50


def compute_cirr_val_metrics_text_image(
    relative_val_dataset: CIRRDataset,
    blip_text_encoder: torch.nn.Module,
    multiple_text_index_features: List[torch.Tensor],
    multiple_text_index_names: List[List[str]],
    image_index_features: torch.Tensor,
    image_index_names: List[str],
    combining_function: callable,
    beta: float
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Compute validation metrics on CIRR dataset combining text and image similarities.

    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param blip_text_encoder: BLIP model
    :param multiple_text_index_features: validation index features from text
    :param multiple_text_index_names: validation index names from text
    :param image_index_features: validation image index features
    :param image_index_names: validation image index names
    :param combining_function: function that combines features
    :param beta: weight for combining text and image similarities
    :return: the computed validation metrics
    """
    all_text_similarities = []
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
            text_features
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
            image_index_features
        )

        # Normalize and compute similarities
        image_index_features = F.normalize(image_index_features, dim=-1).float()
        image_similarities = predicted_image_features @ image_index_features.T
    else:
        image_similarities = torch.zeros_like(all_text_similarities[0])

    # Merge text similarities
    merged_text_similarities = torch.mean(torch.stack(all_text_similarities), dim=0)

    # Merge text and image similarities
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
    print(f"[{datetime.now()}] Compute subset predictions and ground-truth labels")
    group_members = np.array(group_members)
    print(f"[{datetime.now()}] Compute group_mask")
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    print(f"[{datetime.now()}] Compute group_labels")
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    print(f"[{datetime.now()}] Compute assert torch.equal")
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    print(f"[{datetime.now()}] Compute metrics")
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

    return group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50


def generate_cirr_val_predictions(
    blip_text_encoder: torch.nn.Module,
    relative_val_dataset: CIRRDataset,
    combining_function: callable,
    index_names: List[str],
    index_features: torch.Tensor,
    no_print_output: bool = False
) -> Tuple[torch.Tensor, List[str], List[str], List[List[str]]]:
    """
    Compute CIRR predictions on the validation set
    :param blip_text_encoder: BLIP model for text
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param index_features: validation index features
    :param index_names: validation index names
    :param no_print_output: whether to print the output
    :return: predicted features, reference names, target names and group members
    """
    relative_val_loader = DataLoader(
        dataset=relative_val_dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize predicted features, target_names, group_members and reference_names
    predicted_features = torch.empty((0, 256)).to(device, non_blocking=True)
    target_names = []
    group_members = []
    reference_names = []

    if not no_print_output:
        relative_val_loader = tqdm(relative_val_loader)

    for batch_reference_names, batch_target_names, captions, batch_group_members in relative_val_loader:
        # text_inputs = clip.tokenize(captions).to(device, non_blocking=True)
        batch_group_members = np.array(batch_group_members).T.tolist()

        # Compute the predicted features
        with torch.no_grad():
            text_features = blip_text_encoder(captions, max_length=77, device=device)
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if text_features.shape[0] == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(
                    itemgetter(*batch_reference_names)(name_to_feat)
                )
                # To avoid unnecessary computation,
                # retrieve the reference image features directly from the index features
            batch_predicted_features = combining_function(reference_image_features, text_features)

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        target_names.extend(batch_target_names)
        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)

    return predicted_features, reference_names, target_names, group_members


def cirr_val_retrieval(
    combining_function: callable,
    blip_text_encoder: torch.nn.Module,
    blip_img_encoder: torch.nn.Module,
    preprocess: callable
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Perform retrieval on CIRR validation set computing the metrics. To combine the features, the `combining_function` is used
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined features
    :param blip_text_encoder: BLIP text model
    :param blip_img_encoder: BLIP image model
    :param preprocess: preprocess pipeline
    """
    blip_text_encoder = blip_text_encoder.float().eval()
    blip_img_encoder = blip_img_encoder.float().eval()

    # Define the validation datasets and extract the index features
    classic_val_dataset = CIRRDataset('val', 'classic', preprocess)
    index_features, index_names = extract_index_features(classic_val_dataset, blip_img_encoder)
    relative_val_dataset = CIRRDataset('val', 'relative', preprocess)

    return compute_cirr_val_metrics(
        relative_val_dataset,
        blip_text_encoder,
        index_features,
        index_names,
        combining_function
    )


def cirr_val_retrieval_text_image(
    combining_function: callable,
    blip_text_encoder: torch.nn.Module,
    blip_img_encoder: torch.nn.Module,
    text_captions: List[dict],
    beta: float,
    preprocess: callable,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Perform retrieval on CIRR validation set computing the metrics.
    To combine the features, the `combining_function` is used

    :param combining_function: function which takes as input (image_features, text_features) and
                               outputs the combined features
    :param blip_text_encoder: BLIP text model
    :param blip_img_encoder: BLIP image model
    :param text_captions: text captions for the CIRR dataset
    :param beta: weight for combining text and image similarities
    :param preprocess: preprocess pipeline
    :return: the computed validation metrics
    """
    blip_text_encoder = blip_text_encoder.float().eval()
    blip_img_encoder = blip_img_encoder.float().eval()

    # Define the validation datasets and extract the index features
    classic_val_dataset = CIRRDataset('val', 'classic', preprocess)

    multiple_index_features, multiple_index_names = [], []

    for i in range(3):
        index_features, index_names, _ = extract_index_features_with_text_captions(
            classic_val_dataset,
            blip_text_encoder,
            text_captions,
            i + 1
        )
        multiple_index_features.append(index_features)
        multiple_index_names.append(index_names)

    image_index_features, image_index_names = extract_index_features(classic_val_dataset, blip_img_encoder)

    relative_val_dataset = CIRRDataset('val', 'relative', preprocess)

    return compute_cirr_val_metrics_text_image(
        relative_val_dataset,
        blip_text_encoder,
        multiple_index_features,
        multiple_index_names,
        image_index_features,
        image_index_names,
        combining_function,
        beta,
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")

    parser.add_argument("--combining-function", type=str, required=True,
                        help="Which combining function use, should be in ['combiner', 'sum']")

    parser.add_argument("--blip-vit", default='base', type=str,
                        help="BLIP model variant, should be in ['base', 'large']")

    parser.add_argument("--blip-pretrained-path", default='models/model_base.pth', type=str,
                        help="path of the BLIP pretrained model weights")
    parser.add_argument("--med-config-path", default='src/blip_modules/med_config.json', type=str,
                        help="path of the BLIP text encoder med_config.json")
    parser.add_argument("--combiner-path", type=str, help="path to trained Combiner")

    parser.add_argument("--projection-dim", default=640 * 4, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden-dim", default=640 * 8, type=int, help="Combiner hidden dim")

    parser.add_argument("--blip-model-path", type=Path, help="Path to the fine-tuned CLIP model")

    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument("--input-dim", default=384, type=int,
                        help="Input dimension for image transform. Default: inherited from clip_model.visual.input_resolution")
    parser.add_argument("--feature-dim", default=256, type=int,
                        help="Feature dimension as input to combiner. Default: inherited from clip_model.visual.output_dim")
    parser.add_argument("--text_captions_path", type=str, help="Path to the text captions for FashionIQ dataset")
    parser.add_argument('--alpha', default=-1, type=float,
                        help='Weight for combining text and image features use element wise sum if set to 0 ~ 1')
    parser.add_argument("--beta", default=0.5, type=float,
                        help="Weight for combining text and image similarities, Close to 1 gives more weight to text")

    args = parser.parse_args()

    from blip_modules.blip_text_encoder import BLIPTextEncoder
    blip_text_encoder = BLIPTextEncoder(
        args.blip_pretrained_path,
        args.med_config_path,
        use_pretrained_proj_layer=True,
        vit=args.blip_vit
    )  # create BLIP text encoder, load pre-trained checkpoint
    blip_text_encoder = blip_text_encoder.to(device)
    print("blip text encoder loaded.")
    blip_text_encoder.eval()

    from blip_modules.blip_img_encoder import BLIPImgEncoder
    blip_img_encoder = BLIPImgEncoder(
        args.blip_pretrained_path,
        args.blip_vit
    )  # create BLIP text encoder, load pre-trained checkpoint
    blip_img_encoder = blip_img_encoder.to(device)
    print("blip img encoder loaded.")
    blip_img_encoder = blip_img_encoder.eval()

    if args.blip_model_path:
        print('Trying to load the fine-tuned BLIP model')
        state_dict = torch.load(args.blip_model_path, map_location=device)
        blip_text_encoder.load_state_dict(state_dict["BLIPTextEncoder"])
        print('BLIP model loaded successfully')
        print(f"load epoch {state_dict['epoch']}")

    preprocess = None

    if args.transform == 'targetpad':
        print('Target pad preprocess pipeline is used')
        preprocess = targetpad_transform(args.target_ratio, args.input_dim)
    elif args.preprocess == 'squarepad':
        print('Square pad preprocess pipeline is used')
        preprocess = squarepad_transform(args.input_dim)
    else:
        pass

    if args.combining_function.lower() == 'sum':
        if args.combiner_path:
            warnings.warn(
                "Be careful, you are using the element-wise sum as combining_function but you have also passed a path"
                " to a trained Combiner. Such Combiner will not be used"
            )

        if 1 >= args.alpha >= 0:
            combining_function = lambda image_features, text_features: element_wise_sum_with_alpha(
                image_features,
                text_features,
                args.alpha
            )
        else:
            combining_function = element_wise_sum
    elif args.combining_function.lower() == 'combiner':
        combiner = Combiner(args.feature_dim, args.projection_dim, args.hidden_dim).to(device, non_blocking=True)
        state_dict = torch.load(args.combiner_path, map_location=device)
        combiner.load_state_dict(state_dict["Combiner"])
        combiner.eval()
        combining_function = combiner.combine_features
    else:
        raise ValueError("combiner_path should be in ['sum', 'combiner']")

    if args.dataset.lower() == 'cirr':
        if args.text_captions_path:
            with open(args.text_captions_path, 'r') as f:
                text_captions = json.load(f)

            print('Running CIRR validation with text and image similarities combined with beta =', args.beta)

            group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = \
                cirr_val_retrieval_text_image(
                    combining_function,
                    blip_text_encoder,
                    blip_img_encoder,
                    text_captions,
                    args.beta,
                    preprocess
                )
        else:
            group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = \
                cirr_val_retrieval(combining_function, blip_text_encoder, blip_img_encoder, preprocess)

        print(f"{group_recall_at1 = }")
        print(f"{group_recall_at2 = }")
        print(f"{group_recall_at3 = }")
        print(f"{recall_at1 = }")
        print(f"{recall_at5 = }")
        print(f"{recall_at10 = }")
        print(f"{recall_at50 = }")

    elif args.dataset.lower() == 'fashioniq':
        average_recall10_list = []
        average_recall50_list = []

        if args.text_captions_path:
            with open(args.text_captions_path, 'r') as f:
                text_captions = json.load(f)

            print('Running FashionIQ validation with text and image similarities combined with beta =', args.beta)

            shirt_recallat10, shirt_recallat50 = fashioniq_val_retrieval_text_image(
                'shirt',
                combining_function,
                blip_text_encoder,
                blip_img_encoder,
                text_captions,
                args.beta,
                preprocess
            )
            average_recall10_list.append(shirt_recallat10)
            average_recall50_list.append(shirt_recallat50)

            dress_recallat10, dress_recallat50 = fashioniq_val_retrieval_text_image(
                'dress',
                combining_function,
                blip_text_encoder,
                blip_img_encoder,
                text_captions,
                args.beta,
                preprocess
            )

            average_recall10_list.append(dress_recallat10)
            average_recall50_list.append(dress_recallat50)

            toptee_recallat10, toptee_recallat50 = fashioniq_val_retrieval_text_image(
                'toptee',
                combining_function,
                blip_text_encoder,
                blip_img_encoder,
                text_captions,
                args.beta,
                preprocess
            )
            average_recall10_list.append(toptee_recallat10)
            average_recall50_list.append(toptee_recallat50)
        else:
            shirt_recallat10, shirt_recallat50 = fashioniq_val_retrieval(
                'shirt',
                combining_function,
                blip_text_encoder,
                blip_img_encoder,
                preprocess
            )
            average_recall10_list.append(shirt_recallat10)
            average_recall50_list.append(shirt_recallat50)

            dress_recallat10, dress_recallat50 = fashioniq_val_retrieval(
                'dress', combining_function,
                blip_text_encoder,
                blip_img_encoder,
                preprocess
            )
            average_recall10_list.append(dress_recallat10)
            average_recall50_list.append(dress_recallat50)

            toptee_recallat10, toptee_recallat50 = fashioniq_val_retrieval(
                'toptee', combining_function,
                blip_text_encoder,
                blip_img_encoder,
                preprocess
            )
            average_recall10_list.append(toptee_recallat10)
            average_recall50_list.append(toptee_recallat50)

        print(f"\n{shirt_recallat10 = }")
        print(f"{shirt_recallat50 = }")

        print(f"{dress_recallat10 = }")
        print(f"{dress_recallat50 = }")

        print(f"{toptee_recallat10 = }")
        print(f"{toptee_recallat50 = }")

        print(f"average recall10 = {mean(average_recall10_list)}")
        print(f"average recall50 = {mean(average_recall50_list)}")
    else:
        raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ")


if __name__ == '__main__':
    main()
