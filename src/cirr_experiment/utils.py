from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns


def convert_to_pivot_cirr(
    data: List[pd.DataFrame]
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame
]:
    """
    Convert a list of DataFrames to a single DataFrame with a pivot table structure.

    :param data: list of DataFrames
    :return: a single DataFrame with a pivot table structure
    """
    concatenated_data = []
    alphas = np.arange(0, 1.05, 0.05)

    # Concatenate dataframes for all alphas and add alpha_index, alpha_df in enumerate(data):
    for alpha_index, alpha_df in enumerate(data):
        alpha_df['alpha'] = np.round(alphas[alpha_index], 2)
        concatenated_data.append(alpha_df)

    # Create one dataframe per category
    category_df = pd.concat(concatenated_data, ignore_index=True)

    # Create pivot tables for
    recall_at1 = category_df.pivot(index='beta', columns='alpha', values='recall_at1')
    recall_at5 = category_df.pivot(index='beta', columns='alpha', values='recall_at5')
    recall_at10 = category_df.pivot(index='beta', columns='alpha', values='recall_at10')
    recall_at50 = category_df.pivot(index='beta', columns='alpha', values='recall_at50')
    group_recall_at1 = category_df.pivot(index='beta', columns='alpha', values='group_recall_at1')
    group_recall_at2 = category_df.pivot(index='beta', columns='alpha', values='group_recall_at2')
    group_recall_at3 = category_df.pivot(index='beta', columns='alpha', values='group_recall_at3')

    return recall_at1, recall_at5, recall_at10, recall_at50, group_recall_at1, group_recall_at2, group_recall_at3


def prepare_and_plot_recall_pivot(data: pd.DataFrame, title: str):
    """
    Prepare and plot a pivot table for recall@10 or recall@50.

    :param data: Pivot table data
    :param title: plot title
    """
    # Convert index and columns to formatted strings
    data.index = [f"{float(idx):.2f}" for idx in data.index]
    data.columns = [f"{float(col):.2f}" for col in data.columns]

    plt.figure(figsize=(20, 20))
    sns.heatmap(data, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'format': '%.2f'})
    plt.title(title)
    plt.xlabel('Alpha')
    plt.ylabel('Beta')

    # Since we have converted indices and columns to strings, they should display correctly.
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()


def filter_data_by_scale(data: pd.DataFrame, scale: float):
    """
    Filter the data to include only rows and columns that match the given scale (formatted to 2 decimal places).

    :param data: Original data
    :param scale: The scale to filter by (e.g., 0.1)
    :return: Filtered data
    """
    # Convert index and columns to formatted strings
    data.index = [f"{float(idx):.2f}" for idx in data.index]
    data.columns = [f"{float(col):.2f}" for col in data.columns]

    # Create a list of formatted strings that match the desired scale
    scale_values = [f"{i * scale:.2f}" for i in range(int(1 / scale) + 1)]

    # Filter rows and columns by the formatted scale values
    filtered_data = data.loc[data.index.isin(scale_values), data.columns.isin(scale_values)]

    return filtered_data


def filter_and_plot_comparison(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    title1: str,
    title2: str,
    font_size: int = 14,
    annot_font_size: int = 10,
    filter_scale: float = 0.1,
    cmap: str = "magma",
    vmin: float = 0,
    vmax: float = 100,
):
    """
    Filter two datasets and plot them side by side for comparison.

    :param data1: First pivot table data
    :param data2: Second pivot table data
    :param title1: Plot title for the first dataset
    :param title2: Plot title for the second dataset
    :param font_size: Font size for the titles and axis labels
    :param annot_font_size: Font size for the annotations in the heatmaps
    :param filter_scale: Scale to filter the data by
    :param cmap: Color map for the heatmaps
    :param vmin: Minimum value for the color scale
    :param vmax: Maximum value for the color scale
    """
    data1 = filter_data_by_scale(data1, filter_scale)
    data2 = filter_data_by_scale(data2, filter_scale)

    # Convert indices and columns to formatted strings
    data1.index = [f"{float(idx):.2f}" for idx in data1.index]
    data1.columns = [f"{float(col):.2f}" for col in data1.columns]

    data2.index = [f"{float(idx):.2f}" for idx in data2.index]
    data2.columns = [f"{float(col):.2f}" for col in data2.columns]

    # Create figure with GridSpec to control layout
    fig = plt.figure(figsize=(16, 8), dpi=300)
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], figure=fig)

    # Plot first heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(
        data1,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        annot_kws={"size": annot_font_size},
        ax=ax1
    )
    ax1.set_title(title1, fontsize=font_size * 1.5)
    ax1.set_xlabel('Alpha', fontsize=font_size)
    ax1.set_ylabel('Beta', fontsize=font_size)
    ax1.tick_params(axis='x', rotation=45, labelsize=font_size)
    ax1.tick_params(axis='y', rotation=0, labelsize=font_size)

    # Plot second heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(
        data2,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar=True,
        cbar_ax=fig.add_subplot(gs[0, 2]),
        cbar_kws={'format': '%.2f', 'shrink': 0.8},
        annot_kws={"size": annot_font_size}, ax=ax2
    )
    ax2.set_title(title2, fontsize=font_size * 1.5)
    ax2.set_xlabel('Alpha', fontsize=font_size)
    ax2.set_ylabel('Beta', fontsize=font_size)
    ax2.tick_params(axis='x', rotation=45, labelsize=font_size)
    ax2.tick_params(axis='y', rotation=0, labelsize=font_size)

    plt.savefig("comparison_heatmap.png", bbox_inches='tight', dpi=300)
    plt.show()


def prepare_ground_truths(json_data):
    """
    Prepare ground truth data from the JSON structure.

    :param json_data: JSON data containing target and candidate matches with captions
    :return: Dictionary mapping targets to lists of tuples (candidates and captions)
    """
    ground_truths = {}
    for entry in json_data:
        target_hard = entry['target_hard']
        reference = entry['reference']
        caption = entry['caption']
        img_set = entry['img_set']
        if target_hard not in ground_truths:
            ground_truths[target_hard] = []
        ground_truths[target_hard].append((reference, caption, img_set))
    return ground_truths


def plot_retrieval_results_of_i(
    sorted_indices: torch.Tensor,
    image_index_names: List[str],
    target_names: List[str],
    ground_truths: dict,
    top_k: int = 5,
    i: int = 0,
    directory='../../../cirr_dataset/dev/'
):
    """
    Plot retrieval results for a specific query showing the query and its top retrieved images, highlighting ground truths
    and displaying associated captions to the left of the query image.

    :param sorted_indices: 2D tensor or array with sorted indices of retrieved images per query.
    :param image_index_names: List of image paths corresponding to indices in sorted_indices.
    :param target_names: List of names or descriptions for each query.
    :param ground_truths: Dictionary mapping target names to lists of tuples (candidates and captions).
    :param top_k: Number of top retrieved results to display per query.
    :param i: Index of the query to display.
    :param directory: Directory where the images are stored.
    """
    query_index = i
    retrival_result_images = [image_index_names[j] for j in sorted_indices[query_index]][:top_k]
    ground_truth_target = target_names[query_index]
    query_img_path, query_caption, img_set = ground_truths.get(ground_truth_target, [('', '', '')])[0]

    fig = plt.figure(figsize=(30, 5))
    gs = GridSpec(2, top_k + 2, figure=fig)
    query_img = Image.open(f'{directory}{query_img_path}.png')

    # Subplot for the caption text to the left of the query image
    ax_text = fig.add_subplot(gs[0, 0])
    query_name = query_img_path
    ax_text.text(0.5, 0.5, query_caption, va='center', ha='center')
    ax_text.axis('off')

    # Subplot for the query image
    ax_image = fig.add_subplot(gs[0, 1])
    ax_image.imshow(query_img)
    ax_image.set_title(f"Query: {query_name}", color='blue')
    ax_image.axis('off')

    for j, img_path in enumerate(retrival_result_images):
        img = Image.open(f'{directory}{img_path}.png')

        ax_result = fig.add_subplot(gs[0, j + 2])
        ax_result.imshow(img)

        # Check if this image is a ground truth and retrieve its captions
        if img_path == ground_truth_target:
            ax_result.set_title(f"{img_path}", color='green')
        else:
            ax_result.set_title(img_path)

        ax_result.axis('off')

    # Reorder based on retrieval_result_images
    img_set_members = [img for img in [image_index_names[j] for j in sorted_indices[query_index]] if
                       img in img_set['members']]

    # Subplot for the caption text to the left of the query image
    ax_text = fig.add_subplot(gs[1, 0])
    query_name = query_img_path
    ax_text.text(0.5, 0.5, query_caption, va='center', ha='center')
    ax_text.axis('off')

    # Subplot for the query image
    ax_image = fig.add_subplot(gs[1, 1])
    ax_image.imshow(query_img)
    ax_image.set_title(f"Query: {query_name}", color='blue')
    ax_image.axis('off')

    for j, img_path in enumerate(img_set_members):
        img = Image.open(f'{directory}{img_path}.png')

        ax_result = fig.add_subplot(gs[1, j + 2])
        ax_result.imshow(img)

        # Check if this image is a ground truth and retrieve its captions
        if img_path == ground_truth_target:
            ax_result.set_title(f"{img_path}", color='green')
        else:
            ax_result.set_title(img_path)

        ax_result.axis('off')

    plt.tight_layout()
    plt.show()


def found_better_than_original(
    sorted_indices_origin: torch.Tensor,
    image_index_names_origin: List[str],
    target_names_origin: List[str],
    sorted_indices: torch.Tensor,
    image_index_names: List[str],
    target_names: List[str],
    top_k: int = 10,
) -> List[int]:
    """
    Return indices of queries where the new retrieval results are better than the original results,
    which based on the rank of the ground truth in the top_k results.
    This is useful to evaluate the performance of a new retrieval method compared to an existing one.

    Args:
        sorted_indices_origin (torch.Tensor): 2D tensor of sorted indices by relevance per query for the original method.
        image_index_names_origin (List[str]): List of image names corresponding to indices in sorted_indices_origin.
        target_names_origin (List[str]): List of names or descriptions for each query for the original method.
        sorted_indices (torch.Tensor): 2D tensor of sorted indices by relevance per query for the new method.
        image_index_names (List[str]): List of image names corresponding to indices in sorted_indices.
        target_names (List[str]): List of target names each query is supposed to retrieve.
        top_k (int): Number of top retrieved results to consider per query.

    Returns:
        List[int]: List of indices where the new method outperforms the original method in terms of the rank of the ground truth target.
    """
    better_indices = []
    total_queries = len(target_names)

    for index in range(total_queries):
        # Find the index of the ground truth in the top_k results of the original method
        new_rank = [image_index_names[i] for i in sorted_indices[index]].index(target_names[index])
        origin_rank = [image_index_names_origin[i] for i in sorted_indices_origin[index]].index(
            target_names_origin[index])

        if new_rank <= top_k < origin_rank:
            better_indices.append(index)

    return better_indices


def element_wise_sum_original(image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
    """
    Normalized element-wise sum of image features and text features
    :param image_features: non-normalized image features
    :param text_features: non-normalized text features
    :return: normalized element-wise sum of image and text features
    """
    return F.normalize(image_features + text_features, dim=-1)


def element_wise_sum(image_features: torch.Tensor, text_features: torch.Tensor, alpha=0.65) -> torch.Tensor:
    """
    Normalized element-wise sum of image features and text features
    :param image_features: non-normalized image features
    :param text_features: non-normalized text features
    :param alpha: weight for text features
    :return: normalized element-wise sum of image and text features
    """
    return F.normalize((1 - alpha) * image_features + alpha * text_features, dim=-1)


def get_combing_function_with_alpha(alpha: float):
    """
    Get a combing function with a specific beta value.

    :param alpha: weight for text features
    :return: combing function with a specific beta value
    """
    return lambda image_features, text_features: element_wise_sum(image_features, text_features, alpha=alpha)
