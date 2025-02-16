import plotly.graph_objects as go
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from image_drift_generator.image_generator import ImageDatasetGenerator
from image_drift_generator.models import TransformInfo
from tqdm import tqdm

COLORS = [
    '#636EFA',
    '#EF553B',
    '#00CC96',
    '#AB63FA',
    '#FFA15A',
    '#19D3F3',
    '#FF6692',
    '#B6E880',
    '#FF97FF',
    '#FECB52',
    '#FF6E4A',
    '#FFCC96',
    '#B19CD9',
    '#FF6692',
    '#666666',
    '#882E72',
    '#1965B0',
    '#7FC97F',
    '#5B5B5B',
]


def plot_histogram(image, ax, title='Pixel Intensity Histogram'):
    """Plot histogram of pixel intensities for an image."""
    # Image is assumed to be a tensor in CHW format
    image = image.reshape(-1, 3)  # Flatten the image pixels by channel

    colors = ['r', 'g', 'b']  # Color channels
    for i, color in enumerate(colors):
        ax.hist(
            image[:, i],
            bins=256,
            color=color,
            alpha=0.4,
            label=f'Channel {color.upper()}',
            density=False,
        )
    ax.legend()

    ax.set_title(title)
    ax.set_xlim([0, 1])  # Assuming normalized [0, 1] pixel values
    ax.set_ylim([0, 1000])


def plot_similarity_metric(
    df: pl.DataFrame,
    metric_name: str,
    labels: list[str],
    title='',
    width: int = 1200,
    height: int = 800,
) -> None:
    """Plot similarity metric for a specific metric in the given DataFrame.

    Args:
        df (pl.DataFrame): DataFrame containing the data to plot.
        metric_name (str): Name of the column containing the metric to plot.
        labels (list[str]): List of column names to use as labels for the similarity metric.
        title (str, optional): Title of the plot. Defaults to ''.
        width (int, optional): Width of the plot. Defaults to 1200.
        height (int, optional): Height of the plot. Defaults to 800.
    """
    fig = go.Figure()
    # Iterate over rows in the DataFrame
    for i, row in enumerate(df.iter_rows(named=True)):
        # Extract values from the row
        values = row[metric_name]
        sequence_index = np.arange(len(values))

        # Compute confidence intervals (Here we simulate using standard deviation)
        ci = 1.96  # 95% confidence interval
        std_dev = np.std(values) * ci
        lower_bound = np.array(values) - std_dev
        upper_bound = np.array(values) + std_dev

        # Generate label string from the `labels` parameter
        label_str = ', '.join([f'{label} = {row[label]}' for label in labels])

        # Set a color for the line trace (cycle through colors list)
        color = COLORS[i % len(COLORS)]

        # Plot the main line trace
        line_trace = go.Scatter(
            x=sequence_index,
            y=values,
            mode='lines',
            name=label_str,
            line=dict(color=color),  # Set the color explicitly
            legendgroup=f'group_{i}',  # Group the line with its CI
        )

        # Add the main line trace to the figure
        fig.add_trace(line_trace)

        # Convert hex color to rgba with transparency for the CI fillcolor
        r, g, b = tuple(
            int(color.lstrip('#')[j : j + 2], 16) for j in (0, 2, 4)
        )
        fill_color = f'rgba({r}, {g}, {b}, 0.2)'

        # Add the confidence interval trace with the same color but transparent
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([sequence_index, sequence_index[::-1]]),
                y=np.concatenate([upper_bound, lower_bound[::-1]]),
                fill='toself',
                fillcolor=fill_color,  # Set fill color with transparency
                line=dict(color='rgba(255,255,255,0)'),  # No line for the CI
                showlegend=False,  # Do not show CI in legend
                legendgroup=f'group_{i}',
            )
        )

    # Update the layout of the plot
    fig.update_layout(
        title=title,
        xaxis_title='Sequence index',
        yaxis_title=metric_name,
        legend_title='Parameters',
        hovermode='x',
        template='plotly_white',
        legend=dict(
            font=dict(size=10),  # Reduce legend font size
        ),
        width=width,
        height=height,
    )

    # Show the plot
    fig.show()


def plot_boxplot_metric(
    df: pl.DataFrame,
    metric_name: str,
    labels: list[str],
    title='',
    width: int = 1200,
    height: int = 800,
) -> None:
    """Plot box plots for a specific metric in the given DataFrame.

    Args:
        df (pl.DataFrame): DataFrame containing the data to plot.
        metric_name (str): Name of the column containing the metric to plot.
        labels (list[str]): List of column names to use as labels for the box plots.
        title (str, optional): Title of the plot. Defaults to ''.
        width (int, optional): Width of the plot. Defaults to 1200.
        height (int, optional): Height of the plot. Defaults to 800.
    """
    fig = go.Figure()

    # Iterate over rows in the DataFrame
    for i, row in enumerate(df.iter_rows(named=True)):
        # Extract values from the row
        values = row[metric_name]

        # Generate label string from the `labels` parameter
        label_str = ', '.join([f'{label} = {row[label]}' for label in labels])

        # Set a color for the box trace (cycle through colors list)
        color = COLORS[i % len(COLORS)]

        # Plot the box plot for the row
        box_trace = go.Box(
            y=values,
            name=label_str,
            marker_color=color,  # Set the color explicitly
            boxmean='sd',  # Option to show mean and standard deviation
        )

        # Add the box plot trace to the figure
        fig.add_trace(box_trace)

    # Update the layout of the plot
    fig.update_layout(
        title=title,
        xaxis_title='Parameters',
        yaxis_title=metric_name,
        legend_title='Parameters',
        hovermode='closest',
        template='plotly_white',
        width=width,
        height=height,
    )

    # Show the plot
    fig.show()


def visualize_transformations(
    image_generator: ImageDatasetGenerator,
    description: str,
    transform_lists: list[list[TransformInfo]],
):
    for transform_list in tqdm(
        transform_lists,
        desc=description,
        total=len(transform_lists),
    ):
        image_generator.add_abrupt_drift(transform_list)
        for i in range(1):
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            axes[0].imshow(image_generator.dataset[i][0].permute(1, 2, 0))
            axes[1].imshow(
                image_generator.transform_pipeline(
                    image_generator.dataset[i][0]
                ).permute(1, 2, 0)  # type: ignore
            )
            axes[0].set_title('Original Image')
            axes[1].set_title(f'Drift Level: {transform_list[0].drift_level}')
            plt.show()
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            plot_histogram(
                image_generator.dataset[i][0].permute(1, 2, 0),
                axes[0],
                'Original Image Histogram',
            )
            plot_histogram(
                image_generator.transform_pipeline(
                    image_generator.dataset[i][0]
                ).permute(1, 2, 0),  # type: ignore
                axes[1],
                'Transformed Image Histogram',
            )
            plt.show()
