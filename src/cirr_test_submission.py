"""
 * Copyright (c) 2024
 * SPDX-License-Identifier: MIT
 * This file is copied from the Bi-Blip4CIR project and remains under the same license.
 * Original authors and full license text can be found in the LICENSE file at:
 * https://github.com/Cuberick-Orion/Bi-Blip4CIR
 * Original Authors: Cuberick-Orion
"""
import json
import warnings
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import extract_index_features, extract_index_features_with_text_captions
from data_utils import CIRRDataset, targetpad_transform, squarepad_transform, base_path
from combiner import Combiner
from utils import element_wise_sum, device, element_wise_sum_with_alpha


def generate_cirr_test_submissions(
    combining_function: callable,
    file_name: str,
    blip_text_encoder: torch.nn.Module,
    blip_img_encoder: torch.nn.Module,
    preprocess: callable
):
    """
   Generate and save CIRR test submission files to be submitted to evaluation server
   :param combining_function: function which takes as input (image_features, text_features) and outputs the combined features
   :param file_name: file_name of the submission
   :param blip_text_encoder: BLIP model
   :param blip_img_encoder:
   :param preprocess: preprocess pipeline
   """

    blip_text_encoder = blip_text_encoder.float().eval()
    blip_img_encoder = blip_img_encoder.float().eval()

    # Define the dataset and extract index features
    classic_test_dataset = CIRRDataset('test1', 'classic', preprocess)
    index_features, index_names = extract_index_features(classic_test_dataset, blip_img_encoder)
    relative_test_dataset = CIRRDataset('test1', 'relative', preprocess)

    # Generate test prediction dicts for CIRR
    pairid_to_predictions, pairid_to_group_predictions = generate_cirr_test_dicts(
        relative_test_dataset,
        blip_text_encoder,
        index_features,
        index_names,
        combining_function
    )

    submission = {
        'version': 'rc2',
        'metric': 'recall'
    }
    group_submission = {
        'version': 'rc2',
        'metric': 'recall_subset'
    }

    submission.update(pairid_to_predictions)
    group_submission.update(pairid_to_group_predictions)

    # Define the submission path
    submissions_folder_path = base_path / "submission" / 'CIRR'
    submissions_folder_path.mkdir(exist_ok=True, parents=True)

    print(f"Saving CIRR test predictions")
    with open(submissions_folder_path / f"recall_submission_{file_name}.json", 'w+') as file:
        json.dump(submission, file, sort_keys=True)

    with open(submissions_folder_path / f"recall_subset_submission_{file_name}.json", 'w+') as file:
        json.dump(group_submission, file, sort_keys=True)


def generate_cirr_test_submissions_text_image(
    combining_function: callable,
    file_name: str,
    blip_text_encoder: torch.nn.Module,
    blip_img_encoder: torch.nn.Module,
    text_captions: List[dict],
    beta: float,
    preprocess: callable,
):
    """
    Generate and save CIRR test submission files to be submitted to evaluation server

    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined features
    :param file_name: file_name of the submission
    :param blip_text_encoder: BLIP text model
    :param blip_img_encoder: BLIP image model
    :param text_captions: text captions for CIRR dataset
    :param beta: weight for combining text and image similarities
    :param preprocess: preprocess pipeline
    """
    blip_text_encoder = blip_text_encoder.float().eval()
    blip_img_encoder = blip_img_encoder.float().eval()

    # Define the dataset and extract index features
    classic_test_dataset = CIRRDataset('test1', 'classic', preprocess)

    multiple_index_features, multiple_index_names = [], []

    for i in range(3):
        index_features, index_names, _ = extract_index_features_with_text_captions(
            classic_test_dataset,
            blip_text_encoder,
            text_captions,
            i + 1
        )
        multiple_index_features.append(index_features)
        multiple_index_names.append(index_names)

    image_index_features, image_index_names = extract_index_features(classic_test_dataset, blip_img_encoder)

    relative_test_dataset = CIRRDataset('test1', 'relative', preprocess)

    # Generate test prediction dicts for CIRR
    pairid_to_predictions, pairid_to_group_predictions = generate_cirr_test_dicts_text_image(
        relative_test_dataset,
        blip_text_encoder,
        multiple_index_features,
        multiple_index_names,
        image_index_features,
        image_index_names,
        combining_function,
        beta,
    )

    submission = {
        'version': 'rc2',
        'metric': 'recall'
    }
    group_submission = {
        'version': 'rc2',
        'metric': 'recall_subset'
    }

    submission.update(pairid_to_predictions)
    group_submission.update(pairid_to_group_predictions)

    # Define the submission path
    submissions_folder_path = base_path / "submission" / 'CIRR'
    submissions_folder_path.mkdir(exist_ok=True, parents=True)

    print(f"Saving CIRR test predictions")
    with open(submissions_folder_path / f"recall_submission_{file_name}.json", 'w+') as file:
        json.dump(submission, file, sort_keys=True)

    with open(submissions_folder_path / f"recall_subset_submission_{file_name}.json", 'w+') as file:
        json.dump(group_submission, file, sort_keys=True)

def generate_cirr_test_dicts(
    relative_test_dataset: CIRRDataset,
    blip_model: torch.nn.Module,
    index_features: torch.tensor,
    index_names: List[str],
    combining_function: callable
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Compute test prediction dicts for CIRR dataset
    :param relative_test_dataset: CIRR test dataset in relative mode
    :param blip_model: blip_text_encoder
    :param index_features: test index features
    :param index_names: test index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined features
    :return: Top50 global and Top3 subset prediction for each query (reference_name, caption)
    """

    # Generate predictions
    predicted_features, reference_names, group_members, pairs_id = \
        generate_cirr_test_predictions(
            blip_model,
            relative_test_dataset,
            combining_function,
            index_names,
            index_features
        )

    print(f"Compute CIRR prediction dicts")

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the similarities and sort the results
    similarities = predicted_features @ index_features.T
    sorted_indices = torch.argsort(similarities, dim=-1, descending=True).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(
            np.array(reference_names), len(index_names)
        ).reshape(len(sorted_index_names), -1)
    )
    sorted_index_names = sorted_index_names[reference_mask].reshape(
        sorted_index_names.shape[0],
        sorted_index_names.shape[1] - 1
    )
    # Compute the subset predictions
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)

    # Generate prediction dicts
    pairid_to_predictions = {
        str(int(pair_id)): prediction[:50].tolist() for (pair_id, prediction) in zip(pairs_id, sorted_index_names)
    }
    pairid_to_group_predictions = {
        str(int(pair_id)): prediction[:3].tolist() for (pair_id, prediction) in zip(pairs_id, sorted_group_names)
    }

    return pairid_to_predictions, pairid_to_group_predictions


def generate_cirr_test_dicts_text_image(
    relative_test_dataset: CIRRDataset,
    blip_text_encoder: torch.nn.Module,
    multiple_index_features: List[torch.Tensor],
    multiple_index_names: List[List[str]],
    image_index_features: torch.Tensor,
    image_index_names: List[str],
    combining_function: callable,
    beta: float,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Compute test prediction dicts for CIRR dataset
    :param relative_test_dataset: CIRR test dataset in relative mode
    :param blip_text_encoder: blip_text_encoder
    :param multiple_index_features: list of test index features
    :param multiple_index_names: list of test index names
    :param image_index_features: test image index features
    :param image_index_names: test image index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined features
    :param beta: weight for combining text and image similarities
    :return: Top50 global and Top3 subset prediction for each query (reference_name, caption)
    """
    all_text_similarities = []
    reference_names = None
    group_members = None
    pairs_id = None

    # Compute similarities for individual text features
    for text_features, text_names in zip(multiple_index_features, multiple_index_names):
        # Generate text predictions and normalize features
        predicted_text_features, reference_names, group_members, pairs_id = generate_cirr_test_predictions(
            blip_text_encoder,
            relative_test_dataset,
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
        predicted_image_features, _, _, _ = generate_cirr_test_predictions(
            blip_text_encoder,
            relative_test_dataset,
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
        image_index_names if image_index_names else multiple_index_names[0]
    )[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(
            np.array(reference_names),
            len(image_index_names)
        ).reshape(len(sorted_index_names), -1)
    )

    sorted_index_names = sorted_index_names[reference_mask].reshape(
        sorted_index_names.shape[0],
        sorted_index_names.shape[1] - 1
    )

    # Compute the subset predictions
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)

    # Generate prediction dicts
    pairid_to_predictions = {
        str(int(pair_id)): prediction[:50].tolist() for (pair_id, prediction) in zip(pairs_id, sorted_index_names)
    }
    pairid_to_group_predictions = {
        str(int(pair_id)): prediction[:3].tolist() for (pair_id, prediction) in zip(pairs_id, sorted_group_names)
    }

    return pairid_to_predictions, pairid_to_group_predictions

def generate_cirr_test_predictions(
    blip_model: torch.nn.Module,
    relative_test_dataset: CIRRDataset,
    combining_function: callable,
    index_names: List[str],
    index_features: torch.tensor
) -> Tuple[torch.tensor, List[str], List[List[str]], List[str]]:
    """
    Compute CIRR predictions on the test set
    :param blip_model: blip_text_encoder
    :param relative_test_dataset: CIRR test dataset in relative mode
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined features
    :param index_features: test index features
    :param index_names: test index names

    :return: predicted_features, reference_names, group_members and pairs_id
    """
    print(f"Compute CIRR test predictions")

    relative_test_loader = DataLoader(
        dataset=relative_test_dataset,
        batch_size=32,
        num_workers=8,
        pin_memory=True
    )

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize pairs_id, predicted_features, group_members and reference_names
    pairs_id = []
    predicted_features = torch.empty((0, 256)).to(device, non_blocking=True)
    group_members = []
    reference_names = []

    for batch_pairs_id, batch_reference_names, captions, batch_group_members in tqdm(
        relative_test_loader):  # Load data
        # text_inputs = clip.tokenize(captions, context_length=77).to(device)
        batch_group_members = np.array(batch_group_members).T.tolist()

        # Compute the predicted features
        with torch.no_grad():
            text_features = blip_model(captions, max_length=77, device=device)
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if text_features.shape[0] == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unqueeze(0)
            else:
                reference_image_features = torch.stack(
                    itemgetter(*batch_reference_names)(name_to_feat)
                )
                # To avoid unnecessary computation,
                # retrieve the reference image features directly from the index features
            batch_predicted_features = combining_function(reference_image_features, text_features)

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)
        pairs_id.extend(batch_pairs_id)

    return predicted_features, reference_names, group_members, pairs_id


def main():
    parser = ArgumentParser()
    parser.add_argument("--submission-name", type=str, required=True, help="submission file name")

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
    parser.add_argument("--beta", default=0.5, type=float,
                        help="Weight for combining text and image similarities, Close to 1 gives more weight to text")
    parser.add_argument('--alpha', default=-1, type=float,
                        help='Weight for combining text and image features use element wise sum if set to 0 ~ 1')

    args = parser.parse_args()

    from blip_modules.blip_text_encoder import BLIPTextEncoder
    blip_text_encoder = BLIPTextEncoder(
        args.blip_pretrained_path,
        args.med_config_path,
        use_pretrained_proj_layer=True,
        vit=args.blip_vit,
    )  # create BLIP text encoder, load pre-trained checkpoint
    blip_text_encoder = blip_text_encoder.to(device)
    print("blip text encoder loaded.")
    blip_text_encoder.eval()

    from blip_modules.blip_img_encoder import BLIPImgEncoder
    blip_img_encoder = BLIPImgEncoder(
        args.blip_pretrained_path,
        args.blip_vit,
    )  # create BLIP text encoder, load pre-trained checkpoint
    blip_img_encoder = blip_img_encoder.to(device)
    print("blip img encoder loaded.")
    blip_img_encoder.eval()

    input_dim = args.input_dim
    feature_dim = args.feature_dim

    if args.blip_model_path:
        print('Trying to load the fine-tuned BLIP model')
        state_dict = torch.load(args.blip_model_path, map_location=device)
        blip_text_encoder.load_state_dict(state_dict["BLIPTextEncoder"])
        print('BLIP model loaded successfully')
        print(f"load epoch {state_dict['epoch']}")

    preprocess = None

    if args.transform == 'targetpad':
        print('Target pad preprocess pipeline is used')
        preprocess = targetpad_transform(args.target_ratio, input_dim)
    elif args.transform == 'squarepad':
        print('Square pad preprocess pipeline is used')
        preprocess = squarepad_transform(input_dim)
    else:
        warnings.warn(f"transform {args.transform} not recognized")

    if args.combining_function.lower() == 'sum':
        if args.combiner_path:
            warnings.warn(
                "Be careful, you are using the element-wise sum as combining_function but you have also passed a path"
                " to a trained Combiner. Such Combiner will not be used")
        if 1 >= args.alpha >= 0:
            combining_function = lambda image_features, text_features: element_wise_sum_with_alpha(
                image_features,
                text_features,
                args.alpha
            )
        else:
            combining_function = element_wise_sum
    elif args.combining_function.lower() == 'combiner':
        combiner = Combiner(feature_dim, args.projection_dim, args.hidden_dim).to(device)
        saved_state_dict = torch.load(args.combiner_path, map_location=device)
        combiner.load_state_dict(saved_state_dict["Combiner"])
        combiner.eval()
        combining_function = combiner.combine_features
    else:
        raise ValueError("combiner_path should be in ['sum', 'combiner']")

    if args.text_captions_path:
        with open(args.text_captions_path, 'r') as f:
            text_captions = json.load(f)

        print('Running CIRR validation with text and image similarities combined with beta =', args.beta)

        generate_cirr_test_submissions_text_image(
                combining_function,
                args.submission_name,
                blip_text_encoder,
                blip_img_encoder,
                text_captions,
                args.beta,
                preprocess
            )
    else:
        generate_cirr_test_submissions(
            combining_function,
            args.submission_name,
            blip_text_encoder,
            blip_img_encoder,
            preprocess
        )


if __name__ == '__main__':
    main()
