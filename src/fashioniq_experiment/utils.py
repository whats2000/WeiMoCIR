from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec


def convert_to_pivot_fiq(data: List[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert a list of DataFrames to a single DataFrame with a pivot table structure.
    :param data: List of DataFrames
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

    # Create pivot tables for recall@10 and recall@50
    recall_at10 = category_df.pivot_table(index='beta', columns='alpha', values='recall_at10')
    recall_at50 = category_df.pivot_table(index='beta', columns='alpha', values='recall_at50')

    return recall_at10, recall_at50


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


def filter_and_plot_recall_pivot(
    data: pd.DataFrame,
    title: str,
    font_size: int = 16,
    annot_font_size: int = 14,
    filter_scale: float = 0.1,
):
    """
    Prepare and plot a pivot table for recall@10 or recall@50.

    :param data: Pivot table data
    :param title: plot title
    :param font_size: Font size for the title and axis labels
    :param annot_font_size: Font size for the annotations in the heatmap
    :param filter_scale: Scale to filter the data by
    """
    data = filter_data_by_scale(data, filter_scale)

    data.index = [f"{float(idx):.2f}" for idx in data.index]
    data.columns = [f"{float(col):.2f}" for col in data.columns]

    plt.figure(figsize=(8, 8), dpi=300)  # Adjust figure size and resolution
    sns.heatmap(data, annot=True, fmt=".2f", cmap="magma", vmin=0, vmax=100,
                cbar_kws={'format': '%.2f'}, annot_kws={"size": annot_font_size})
    plt.title(title, fontsize=font_size)
    plt.xlabel('Alpha', fontsize=font_size)
    plt.ylabel('Beta', fontsize=font_size)
    plt.xticks(rotation=45, fontsize=font_size)
    plt.yticks(rotation=0, fontsize=font_size)

    plt.tight_layout()
    plt.savefig("heatmap.png", bbox_inches='tight', dpi=300)
    plt.show()


def prepare_ground_truths(json_data) -> dict:
    """
    Prepare ground truth data from the JSON structure.

    :param json_data: JSON data containing target and candidate matches with captions
    :return: Dictionary mapping targets to lists of tuples (candidates and captions)
    """
    ground_truths = {}
    for entry in json_data:
        target = entry['target']
        candidate = entry['candidate']
        captions = entry['captions']
        if target not in ground_truths:
            ground_truths[target] = []
        ground_truths[target].append((candidate, captions))
    return ground_truths


def plot_retrieval_results_of_i(
    sorted_indices: torch.Tensor,
    image_index_names: List[str],
    target_names: List[str],
    ground_truths: dict,
    top_k: int = 5,
    i: int = 0,
    directory='../../../fashionIQ_dataset/images/'
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
    :param directory: Directory containing the images.
    """
    query_index = i
    retrival_result_images = [image_index_names[j] for j in sorted_indices[query_index]][:top_k]
    ground_truth_target = target_names[query_index]
    query_img_path, query_captions = ground_truths.get(ground_truth_target, [('', '')])[0]

    fig = plt.figure(figsize=(30, 5))
    gs = GridSpec(1, top_k + 2, figure=fig)
    query_img = Image.open(f'{directory}{query_img_path}.png')

    # Subplot for the caption text to the left of the query image
    ax_text = fig.add_subplot(gs[0, 0])
    query_name = query_img_path
    query_captions = '\n'.join(sentences for sentences in query_captions)
    ax_text.text(0.5, 0.5, query_captions, va='center', ha='center')
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


def element_wise_sum_original(image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
    """
    Normalized element-wise sum of image features and text features
    :param image_features: non-normalized image features
    :param text_features: non-normalized text features
    :return: normalized element-wise sum of image and text features
    """
    return F.normalize(image_features + text_features, dim=-1)


def element_wise_sum(image_features: torch.tensor, text_features: torch.tensor, alpha=0.65) -> torch.tensor:
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
