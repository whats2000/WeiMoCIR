"""
 * Copyright (c) 2024 whats2000
 * All rights reserved.
 * SPDX-License-Identifier: MIT
 * This file incorporates code from the Bi-Blip4CIR project, available at:
 * https://github.com/Cuberick-Orion/Bi-Blip4CIR
 * For full license text of the original code, see the LICENSE file in the root of the Bi-Blip4CIR repo.
"""
import json
from argparse import ArgumentParser
from datetime import datetime
from operator import itemgetter
from pathlib import Path
from statistics import mean
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from rich import print
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import FashionIQDataset, targetpad_transform, CIRRDataset
from utils import extract_index_features, collate_fn, element_wise_sum, device, \
    extract_index_features_with_text_captions_clip, element_wise_sum_with_beta, extract_index_features_clip


def compute_fiq_val_metrics(relative_val_dataset: FashionIQDataset,
                            clip_text_encoder: torch.nn.Module,
                            clip_tokenizer,
                            index_features: torch.tensor,
                            index_names: List[str],
                            combining_function: callable) -> Tuple[float, float]:
    """
    Compute validation metrics on FashionIQ dataset
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param clip_text_encoder: CLIP model
    :param index_features: validation index features
    :param index_names: validation index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: the computed validation metrics
    """

    # Generate predictions
    predicted_features, target_names = generate_fiq_val_predictions(clip_text_encoder,
                                                                    clip_tokenizer,
                                                                    relative_val_dataset,
                                                                    combining_function, index_names, index_features)

    print(f"[{datetime.now()}] Compute FashionIQ {relative_val_dataset.dress_types} validation metrics")

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    return recall_at10, recall_at50


def compute_fiq_val_metrics_text_image(
    relative_val_dataset: FashionIQDataset,
    clip_text_encoder: torch.nn.Module,
    clip_tokenizer,
    multiple_text_index_features: List[torch.tensor],
    multiple_text_index_names: List[List[str]],
    image_index_features: torch.tensor,
    image_index_names: List[str],
    combining_function: callable,
    alpha: float
) -> Tuple[float, float]:
    """
    Compute validation metrics on FashionIQ dataset combining text and image distances.

    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param clip_text_encoder: CLIP model
    :param clip_tokenizer: CLIP tokenizer
    :param multiple_text_index_features: validation index features from text
    :param multiple_text_index_names: validation index names from text
    :param image_index_features: validation image index features
    :param image_index_names: validation image index names
    :param combining_function: function that combines features
    :param alpha: weight for combining text and image distances
    :return: the computed validation metrics
    """
    all_text_distances = []
    target_names = None

    # Compute distances for individual text features
    for text_features, text_names in zip(multiple_text_index_features, multiple_text_index_names):
        # Generate text predictions and normalize features
        predicted_text_features, target_names = generate_fiq_val_predictions(
            clip_text_encoder,
            clip_tokenizer,
            relative_val_dataset,
            combining_function,
            text_names,
            text_features
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
        predicted_image_features, _ = generate_fiq_val_predictions(clip_text_encoder,
                                                                   clip_tokenizer,
                                                                   relative_val_dataset,
                                                                   combining_function, image_index_names,
                                                                   image_index_features)

        # Normalize and compute distances
        image_index_features = F.normalize(image_index_features, dim=-1).float()
        image_distances = 1 - predicted_image_features @ image_index_features.T
    else:
        image_distances = torch.zeros_like(all_text_distances[0])

    # Merge text distances
    merged_text_distances = torch.mean(torch.stack(all_text_distances), dim=0)

    # Merge text and image distances
    merged_distances = alpha * merged_text_distances + (1 - alpha) * image_distances

    # Sort the results
    sorted_indices = torch.argsort(merged_distances, dim=-1).cpu()
    sorted_index_names = np.array(
        image_index_names if image_index_names else multiple_text_index_names[0]
    )[sorted_indices]

    # Compute the ground-truth labels with respect to the predictions
    labels = torch.tensor(sorted_index_names == np.repeat(
        np.array(target_names),
        len(sorted_index_names[0])).reshape(len(target_names), -1)
                          )

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    return recall_at10, recall_at50


def generate_fiq_val_predictions(clip_text_encoder: torch.nn.Module,
                                 clip_tokenizer,
                                 relative_val_dataset: FashionIQDataset,
                                 combining_function: callable,
                                 index_names: List[str],
                                 index_features: torch.tensor) -> \
    Tuple[torch.tensor, List[str]]:
    """
    Compute FashionIQ predictions on the validation set
    :param clip_text_encoder: CLIP model
    :param clip_tokenizer: CLIP tokenizer
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined features
    :param index_features: validation index features
    :param index_names: validation index names
    :return: predicted features and target names
    """
    print(f"[{datetime.now()}] Compute FashionIQ {relative_val_dataset.dress_types} validation predictions")

    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32,
                                     num_workers=4, pin_memory=False, collate_fn=collate_fn,
                                     shuffle=False)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize predicted features and target names
    predicted_features = torch.empty((0, clip_text_encoder.text_projection.out_features)).to(device, non_blocking=True)
    target_names = []

    for reference_names, batch_target_names, captions in tqdm(relative_val_loader):  # Load data

        # Concatenate the captions in a deterministic way
        flattened_captions: list = np.array(captions).T.flatten().tolist()
        input_captions = [
            f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}" for
            i in range(0, len(flattened_captions), 2)]

        # Compute the predicted features
        with torch.no_grad():
            text_tokens = clip_tokenizer(input_captions, context_length=77).to(device, non_blocking=True)
            text_features = clip_text_encoder(text_tokens, return_dict=False)[0]
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if text_features.shape[0] == 1:
                reference_image_features = itemgetter(*reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*reference_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            batch_predicted_features = combining_function(reference_image_features, text_features)

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        target_names.extend(batch_target_names)

    return predicted_features, target_names


def fashioniq_val_retrieval(dress_type: str,
                            combining_function: callable,
                            clip_text_encoder: torch.nn.Module,
                            clip_img_encoder: torch.nn.Module,
                            preprocess: callable):
    """
    Perform retrieval on FashionIQ validation set computing the metrics. To combine the features the `combining_function`
    is used
    :param dress_type: FashionIQ category on which perform the retrieval
    :param combining_function:function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param clip_text_encoder: CLIP text model
    :param clip_img_encoder: CLIP image model
    :param preprocess: preprocess pipeline
    """

    clip_text_encoder = clip_text_encoder.float().eval()
    clip_img_encoder = clip_img_encoder.float().eval()

    # Define the validation datasets and extract the index features
    classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess)
    index_features, index_names = extract_index_features(classic_val_dataset, clip_img_encoder)
    relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess)

    return compute_fiq_val_metrics(relative_val_dataset,
                                   clip_text_encoder,
                                   clip_img_encoder,
                                   index_features,
                                   index_names,
                                   combining_function)


def fashioniq_val_retrieval_text_image(dress_type: str,
                                       combining_function: callable,
                                       clip_text_encoder: torch.nn.Module,
                                       clip_img_encoder: torch.nn.Module,
                                       clip_tokenizer,
                                       text_captions: List[dict],
                                       alpha: float,
                                       preprocess: callable):
    """
    Perform retrieval on FashionIQ validation set computing the metrics. To combine the features the `combining_function`
    is used
    :param dress_type: FashionIQ category on which perform the retrieval
    :param combining_function:function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param clip_text_encoder: CLIP text model
    :param clip_img_encoder: CLIP image model
    :param clip_tokenizer: CLIP tokenizer
    :param text_captions: text captions for the FashionIQ dataset
    :param alpha: weight for combining text and image distances
    :param preprocess: preprocess pipeline
    """

    clip_text_encoder = clip_text_encoder.float().eval()
    clip_img_encoder = clip_img_encoder.float().eval()

    # Define the validation datasets and extract the index features
    classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess)

    multiple_index_features, multiple_index_names = [], []

    for i in range(3):
        index_features, index_names, _ = extract_index_features_with_text_captions_clip(
            classic_val_dataset,
            clip_text_encoder,
            clip_tokenizer,
            text_captions,
            i + 1
        )
        multiple_index_features.append(index_features)
        multiple_index_names.append(index_names)

    image_index_features, image_index_names = extract_index_features_clip(classic_val_dataset, clip_img_encoder)

    relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess)

    return compute_fiq_val_metrics_text_image(
        relative_val_dataset,
        clip_text_encoder,
        clip_tokenizer,
        multiple_index_features,
        multiple_index_names,
        image_index_features,
        image_index_names,
        combining_function,
        alpha
    )


def compute_cirr_val_metrics(relative_val_dataset: CIRRDataset,
                             clip_model: torch.nn.Module,
                             index_features: torch.tensor,
                             index_names: List[str],
                             combining_function: callable) -> Tuple[
    float, float, float, float, float, float, float]:
    """
    Compute validation metrics on CIRR dataset
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param clip_model: CLIP model
    :param index_features: validation index features
    :param index_names: validation index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: the computed validation metrics
    """
    # Generate predictions
    predicted_features, reference_names, target_names, group_members = \
        generate_cirr_val_predictions(clip_model, relative_val_dataset, combining_function, index_names, index_features)

    print(f"[{datetime.now()}] Compute CIRR validation metrics")

    # Normalize the index features
    print(f"[{datetime.now()}] Compute the index features")
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    print(f"[{datetime.now()}] Compute the distances and sort the results")
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
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


def generate_cirr_val_predictions(clip_model: torch.nn.Module,
                                  relative_val_dataset: CIRRDataset,
                                  combining_function: callable,
                                  index_names: List[str],
                                  index_features: torch.tensor) -> \
    Tuple[torch.tensor, List[str], List[str], List[List[str]]]:
    """
    Compute CIRR predictions on the validation set
    :param clip_model: CLIP model
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param index_features: validation index features
    :param index_names: validation index names
    :return: predicted features, reference names, target names and group members
    """
    print(f"[{datetime.now()}] Compute CIRR validation predictions")
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=4,
                                     pin_memory=True, collate_fn=collate_fn)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize predicted features, target_names, group_members and reference_names
    predicted_features = torch.empty((0, 256)).to(device, non_blocking=True)
    target_names = []
    group_members = []
    reference_names = []

    for batch_reference_names, batch_target_names, captions, batch_group_members in tqdm(
        relative_val_loader):  # Load data
        # text_inputs = clip.tokenize(captions).to(device, non_blocking=True)
        batch_group_members = np.array(batch_group_members).T.tolist()

        # Compute the predicted features
        with torch.no_grad():
            text_features = clip_model(captions, max_length=77, device=device)
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if text_features.shape[0] == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*batch_reference_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            batch_predicted_features = combining_function(reference_image_features, text_features)

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        target_names.extend(batch_target_names)
        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)

    return predicted_features, reference_names, target_names, group_members


def cirr_val_retrieval(combining_function: callable,
                       clip_text_encoder: torch.nn.Module,
                       clip_img_encoder: torch.nn.Module,
                       preprocess: callable):
    """
    Perform retrieval on CIRR validation set computing the metrics. To combine the features the `combining_function`
    is used
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param clip_text_encoder: CLIP text model
    :param clip_img_encoder: CLIP image model
    :param preprocess: preprocess pipeline
    """

    clip_text_encoder = clip_text_encoder.float().eval()
    clip_img_encoder = clip_img_encoder.float().eval()

    # Define the validation datasets and extract the index features
    classic_val_dataset = CIRRDataset('val', 'classic', preprocess)
    index_features, index_names = extract_index_features(classic_val_dataset, clip_img_encoder)
    relative_val_dataset = CIRRDataset('val', 'relative', preprocess)

    return compute_cirr_val_metrics(relative_val_dataset,
                                    clip_text_encoder,
                                    index_features,
                                    index_names,
                                    combining_function)


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")

    parser.add_argument("--clip_name", type=str,
                        help="Name of the CLIP model should be in like ['openai/clip-vit-large-patch14', 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K', 'Geonmo/CLIP-Giga-config-fixed']")

    parser.add_argument("--clip-model-path", type=Path, help="Path to the fine-tuned CLIP model")

    parser.add_argument("--transform", default="clip", type=str, choices=['clip', 'targetpad'],
                        help="Preprocess pipeline to use")

    parser.add_argument("--text_captions_path", type=str, help="Path to the text captions for FashionIQ dataset")
    parser.add_argument("--alpha", default=0.5, type=float,
                        help="Weight for combining text and image distances, Close to 1 gives more weight to text")
    parser.add_argument('--beta', default=-1, type=float,
                        help='Weight for combining text and image features use element wise sum if set to 0 ~ 1')

    args = parser.parse_args()

    from transformers import CLIPTextModelWithProjection

    clip_text_encoder = CLIPTextModelWithProjection.from_pretrained(args.clip_name, torch_dtype=torch.float32)
    clip_text_encoder = clip_text_encoder.float().to(device)

    print("clip text encoder loaded.")
    clip_text_encoder.eval()

    from transformers import CLIPVisionModelWithProjection
    clip_img_encoder = CLIPVisionModelWithProjection.from_pretrained(args.clip_name, torch_dtype=torch.float32)

    clip_img_encoder = clip_img_encoder.float().to(device)
    print("clip img encoder loaded.")
    clip_img_encoder = clip_img_encoder.eval()

    if args.clip_model_path:
        print('Trying to load the fine-tuned CLIP model')
        state_dict = torch.load(args.clip_model_path, map_location=device)
        clip_text_encoder.load_state_dict(state_dict["CLIPTextEncoder"])
        print('CLIP model loaded successfully')
        print(f"load epoch {state_dict['epoch']}")

    preprocess = None

    if args.transform == 'targetpad':
        print('Target pad preprocess pipeline is used')
        preprocess = targetpad_transform(args.target_ratio, args.input_dim)
    elif args.transform == 'clip':
        print('CLIP preprocess pipeline is used')
        from transformers import CLIPImageProcessor
        preprocess = CLIPImageProcessor(
            crop_size={'height': 224, 'width': 224},
            do_center_crop=True,
            do_convert_rgb=True,
            do_normalize=True,
            do_rescale=True,
            do_resize=True,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
            resample=3,
            size={'shortest_edge': 224},
        )
    else:
        pass

    from clip import tokenize

    clip_tokenizer = tokenize

    if 1 >= args.beta >= 0:
        combining_function = lambda image_features, text_features: element_wise_sum_with_beta(
            image_features,
            text_features,
            args.beta
        )
    else:
        combining_function = element_wise_sum

    if args.dataset.lower() == 'cirr':
        raise ValueError("CIRR dataset is not supported yet")

    elif args.dataset.lower() == 'fashioniq':
        average_recall10_list = []
        average_recall50_list = []

        if args.text_captions_path:
            with open(args.text_captions_path, 'r') as f:
                text_captions = json.load(f)

            print('Running FashionIQ validation with text and image distances combined with alpha =', args.alpha)

            shirt_recallat10, shirt_recallat50 = fashioniq_val_retrieval_text_image(
                'shirt',
                combining_function,
                clip_text_encoder,
                clip_img_encoder,
                clip_tokenizer,
                text_captions,
                args.alpha,
                preprocess
            )
            average_recall10_list.append(shirt_recallat10)
            average_recall50_list.append(shirt_recallat50)

            dress_recallat10, dress_recallat50 = fashioniq_val_retrieval_text_image(
                'dress',
                combining_function,
                clip_text_encoder,
                clip_img_encoder,
                clip_tokenizer,
                text_captions,
                args.alpha,
                preprocess
            )

            average_recall10_list.append(dress_recallat10)
            average_recall50_list.append(dress_recallat50)

            toptee_recallat10, toptee_recallat50 = fashioniq_val_retrieval_text_image(
                'toptee',
                combining_function,
                clip_text_encoder,
                clip_img_encoder,
                clip_tokenizer,
                text_captions,
                args.alpha,
                preprocess
            )
            average_recall10_list.append(toptee_recallat10)
            average_recall50_list.append(toptee_recallat50)
        else:
            shirt_recallat10, shirt_recallat50 = fashioniq_val_retrieval('shirt', combining_function,
                                                                         clip_text_encoder, clip_img_encoder,
                                                                         preprocess)
            average_recall10_list.append(shirt_recallat10)
            average_recall50_list.append(shirt_recallat50)

            dress_recallat10, dress_recallat50 = fashioniq_val_retrieval('dress', combining_function,
                                                                         clip_text_encoder, clip_img_encoder,
                                                                         preprocess)
            average_recall10_list.append(dress_recallat10)
            average_recall50_list.append(dress_recallat50)

            toptee_recallat10, toptee_recallat50 = fashioniq_val_retrieval('toptee', combining_function,
                                                                           clip_text_encoder, clip_img_encoder,
                                                                           preprocess)
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

        return (
            shirt_recallat10, shirt_recallat50,
            dress_recallat10, dress_recallat50,
            toptee_recallat10, toptee_recallat50,
            mean(average_recall10_list), mean(average_recall50_list)
        )
    else:
        raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ")


if __name__ == '__main__':
    main()
