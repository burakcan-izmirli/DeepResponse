""" Visualizer """
import logging
import matplotlib.pyplot as plt


def sketch_scatter_plot(true_values, predictions, comet, filename_prefix=""):
    """
    This function creates a scatter plot of true values versus predictions.
    The color of each point represents the squared error between the true value and the prediction.

    Parameters:
    true_values (list): A list of true values.
    predictions (list): A list of predicted values.
    filename_prefix (str): Prefix for the output filename to make it unique.

    Returns:
    None
    """

    # Calculate the squared error for each prediction
    errors = [(true - pred) ** 2 for true, pred in zip(true_values, predictions)]

    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(true_values, predictions, c=errors, cmap='bwr')

    # Add a line of perfect fit
    min_val = min(min(true_values), min(predictions))
    max_val = max(max(true_values), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--')

    # Add labels and title
    plt.xlabel('True Drug Response Values')
    plt.ylabel('Predicted Drug Response Values')
    plt.title('Scatter Plot')

    # Add a color bar
    plt.colorbar(label='Squared Error')

    # Save the plot as a PNG file with dynamic filename
    filename = f'{filename_prefix}scatter_plot.png' if filename_prefix else 'scatter_plot.png'
    plt.savefig(filename)

    if comet is not None:
        comet.log_image(filename)


def sketch_histogram(true_values, predictions, comet, filename_prefix=""):
    """
    This function creates two histograms in the same figure, one for the true values and one for the predictions.

    Parameters:
    true_values (list): A list of true values.
    predictions (list): A list of predicted values.
    filename_prefix (str): Prefix for the output filename to make it unique.

    Returns:
    None
    """

    min_val = min(min(true_values), min(predictions))
    max_val = max(max(true_values), max(predictions))

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 6), tight_layout=True)

    # Create histograms for the true values and predictions
    axs[0].hist(true_values, bins=50, color='crimson', log=True)
    axs[1].hist(predictions, bins=50, color='turquoise', log=True)

    # Add labels and title
    axs[0].set_title('True Drug Response Values')
    axs[1].set_title('Predicted Drug Response Values')
    for ax in axs.flat:
        ax.set(xlabel='pIC50', ylabel='Number of Data Points')

    # Set the axis by looking at the min and max of true_values and predictions
    axs[0].set_xlim(min_val, max_val)
    axs[1].set_xlim(min_val, max_val)

    # Get the maximum of the current y-limits
    max_ylim = max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])

    # Get the minimum non-zero value in the data
    min_ylim = min(value for value in (true_values + predictions) if value > 0)

    # Set the y-limits to be the same
    axs[0].set_ylim(min_ylim, max_ylim)
    axs[1].set_ylim(min_ylim, max_ylim)

    # Save the plot as a PNG file with dynamic filename
    filename = f'{filename_prefix}histogram.png' if filename_prefix else 'histogram.png'
    plt.savefig(filename)

    if comet is not None:
        comet.log_image(filename)


def visualize_results(true_values, predictions, comet, filename_prefix=""):
    """ Main function to visualize the results """
    # Calculate min_val and max_val

    sketch_scatter_plot(true_values, predictions, comet, filename_prefix)
    sketch_histogram(true_values, predictions, comet, filename_prefix)

    logging.info("Visualize the results was performed.")

    return True
