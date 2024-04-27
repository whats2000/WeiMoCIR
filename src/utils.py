"""
 * Copyright (c) 2024 whats2000
 * All rights reserved.
 * SPDX-License-Identifier: MIT
 * This file incorporates code from the Bi-Blip4CIR project, available at:
 * https://github.com/Cuberick-Orion/Bi-Blip4CIR
 * For full license text of the original code, see the LICENSE file in the root of the Bi-Blip4CIR repo.
"""
import random
from pathlib import Path
from typing import Union, Tuple, List

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from rich import print

from data_utils import CIRRDataset, FashionIQDataset

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def extract_index_features(dataset: Union[CIRRDataset, FashionIQDataset], blip_model: nn.Module) -> \
        Tuple[torch.tensor, List[str]]:
    """
    Extract FashionIQ or CIRR index features
    :param dataset: FashionIQ or CIRR dataset in 'classic' mode
    :param blip_model: BLIP model
    :return: a tensor of features and a list of images
    """
    feature_dim = 256
    classic_val_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=4,
                                    pin_memory=False, collate_fn=collate_fn)
    index_features = torch.empty((0, feature_dim)).to(device, non_blocking=True)
    index_names = []
    if isinstance(dataset, CIRRDataset):
        print(f"[{datetime.now()}] extracting CIRR {dataset.split} index features with image features")
    elif isinstance(dataset, FashionIQDataset):
        print(f"[{datetime.now()}] extracting fashionIQ {dataset.dress_types} - {dataset.split} index features with image features")
    
    # original
    for names, images in tqdm(classic_val_loader):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            batch_features = blip_model(images)
            index_features = torch.vstack((index_features, batch_features))
            index_names.extend(names)
    return index_features, index_names

def extract_index_features_with_text_captions(
    dataset: Union[CIRRDataset, FashionIQDataset],
    blip_text_encoder: nn.Module,
    text_captions: List[dict],
    k_th: int = 1
) -> Tuple[torch.tensor, List[str], List[str]]:
    """
    Extract index features using k-th text captions from a list of captions.

    :param dataset: FashionIQ or CIRR dataset in 'classic' mode
    :param blip_text_encoder: BLIP text encoder model
    :param text_captions: The list of text captions dictionaries from Large Language Model
    :param k_th: The k-th text caption to use (1-indexed)
    :return: a tensor of encoded text features, list of image names, and list of used captions
    """

    # Create a mapping from image names to k-th text caption
    candidate_to_caption = {item['candidate']: item['captions'][k_th - 1]
                            for item in text_captions if 'candidate' in item and 'captions' in item}

    # Create DataLoader to iterate over the dataset
    classic_val_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=4, pin_memory=False, collate_fn=collate_fn)

    # Initialize empty tensor for features, and lists for names and captions
    index_features = torch.empty((0, 256)).to(device, non_blocking=True)
    index_names = []
    used_captions = []

    if isinstance(dataset, CIRRDataset):
        print(f"[{datetime.now()}] extracting CIRR {dataset.split} index features with {k_th}-th text captions")
    elif isinstance(dataset, FashionIQDataset):
        print(f"[{datetime.now()}] extracting fashionIQ {dataset.dress_types} - {dataset.split} index features with {k_th}-th text captions")

    # Process each batch
    for names, _ in tqdm(classic_val_loader):
        # Find the corresponding k-th captions using the names
        batch_captions = [candidate_to_caption.get(name, "") for name in names]

        # Encode the captions using BLIP text encoder
        with torch.no_grad():
            # We assume blip_text_encoder can handle empty strings and will return zero vectors for them
            text_features = blip_text_encoder(batch_captions, 77, device)

        # Collect features, names, and captions
        index_features = torch.vstack((index_features, text_features))
        index_names.extend(names)
        used_captions.extend(batch_captions)

    return index_features, index_names, used_captions

def element_wise_sum(image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
    """
    Normalized element-wise sum of image features and text features
    :param image_features: non-normalized image features
    :param text_features: non-normalized text features
    :return: normalized element-wise sum of image and text features
    """
    return F.normalize(image_features + text_features, dim=-1)


def generate_randomized_fiq_caption(flattened_captions: List[str]) -> List[str]:
    """
    Function which randomize the FashionIQ training captions in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1
    (d) cap2
    :param flattened_captions: the list of caption to randomize, note that the length of such list is 2*batch_size since
     to each triplet are associated two captions
    :return: the randomized caption list (with length = batch_size)
    """
    captions = []
    for i in range(0, len(flattened_captions), 2):
        random_num = random.random()
        if random_num < 0.25:
            captions.append(
                f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}")
        elif 0.25 < random_num < 0.5:
            captions.append(
                f"{flattened_captions[i + 1].strip('.?, ').capitalize()} and {flattened_captions[i].strip('.?, ')}")
        elif 0.5 < random_num < 0.75:
            captions.append(f"{flattened_captions[i].strip('.?, ').capitalize()}")
        else:
            captions.append(f"{flattened_captions[i + 1].strip('.?, ').capitalize()}")
    return captions


def collate_fn(batch: list):
    """
    Discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def update_train_running_results(train_running_results: dict, loss: torch.tensor, images_in_batch: int):
    """
    Update `train_running_results` dict during training
    :param train_running_results: logging training dict
    :param loss: computed loss for batch
    :param images_in_batch: num images in the batch
    """
    train_running_results['accumulated_train_loss'] += loss.to('cpu',
                                                               non_blocking=True).detach().item() * images_in_batch
    train_running_results["images_in_epoch"] += images_in_batch


def set_train_bar_description(train_bar, epoch: int, num_epochs: int, train_running_results: dict):
    """
    Update tqdm train bar during training
    :param train_bar: tqdm training bar
    :param epoch: current epoch
    :param num_epochs: numbers of epochs
    :param train_running_results: logging training dict
    """
    train_bar.set_description(
        desc=f"[{epoch}/{num_epochs}] "
             f"train loss: {train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch']:.3f} "
    )


def save_model(name: str, cur_epoch: int, model_to_save: nn.Module, training_path: Path, optimizer):
    """
    Save the weights of the model during training
    :param name: name of the file
    :param cur_epoch: current epoch
    :param model_to_save: pytorch model to be saved
    :param training_path: path associated with the training run
    """
    models_path = training_path / "saved_models"
    models_path.mkdir(exist_ok=True, parents=True)
    model_name = model_to_save.__class__.__name__
    torch.save({
        'epoch': cur_epoch,
        model_name: model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, str(models_path / f'{name}.pt'))


import math
def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr, onlyGroup0=False):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    param_group_count = 0
    for param_group in optimizer.param_groups:
        param_group_count += 1
        if param_group_count <= 1 and onlyGroup0: # only vary group0 parameters' learning rate, i.e., exclude the text_proj layer
            param_group['lr'] = lr
        
def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    

def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):        
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate**epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    