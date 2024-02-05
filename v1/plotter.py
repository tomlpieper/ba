import json
import os
import matplotlib.pyplot as plt

class TrainingMetricsPlotter:
    def __init__(self, json_file):
        with open(json_file, 'r') as file:
            self.data = json.load(file)

    def plot_metrics(self, metric_prefix, save_path=None):
        metric_entries = [entry for entry in self.data['log_history'] if metric_prefix in entry]

        steps = [entry['step'] for entry in metric_entries]
        metrics = [entry[metric_prefix] for entry in metric_entries]

        plt.plot(steps, metrics, marker='o', label=metric_prefix.capitalize())
        plt.title(f'{metric_prefix.capitalize()} Over Steps')
        plt.xlabel('Training Steps')
        plt.ylabel(metric_prefix.capitalize())
        plt.legend()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def plot_multiple_metrics(self, metric_prefixes, save_path=None):
        for prefix in metric_prefixes:
            self.plot_metrics(prefix, save_path=save_path)

# Example usage
json_file_path = '/Volumes/SandigeDisk/code_2/results/eval_metrics.json'
plotter = TrainingMetricsPlotter(json_file_path)
path = "results/stats/"
os.makedirs(path, exist_ok=True)

# Save individual metrics as PNG with legend
plotter.plot_metrics('eval_loss', save_path=path + 'eval_loss_plot.png')
plotter.plot_metrics('train_loss', save_path=path + 'train_loss_plot.png')
plotter.plot_metrics('eval_f1_score', save_path=path + 'eval_f1_score_plot.png')
plotter.plot_metrics('train_f1_score', save_path=path + 'train_f1_score_plot.png')

# Save multiple metrics as one PNG with legend
# plotter.plot_multiple_metrics(['eval_loss', 'train_loss', 'eval_f1_score', 'train_f1_score'], save_path='combined_metrics_plot.png')
