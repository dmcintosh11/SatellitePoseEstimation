import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

def plot_metric_comparison(model_names, metric_values, metric_name, output_dir):
    """
    Generates and saves a bar plot comparing a specific metric across models.
    """
    if not model_names or not metric_values:
        print(f"No data to plot for metric: {metric_name}")
        return

    plt.figure(figsize=(10 + len(model_names) * 0.5, 7))
    
    # Create bar plot
    bar_positions = np.arange(len(model_names))
    bars = plt.bar(bar_positions, metric_values, color=plt.cm.viridis(np.linspace(0, 1, len(model_names))))

    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.title(f'Comparison of Models: {metric_name.replace("_", " ").title()}')
    plt.xticks(bar_positions, model_names, rotation=45, ha="right")
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add text labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * max(metric_values, default=0), f'{yval:.4f}', ha='center', va='bottom')


    plot_filename = f"comparison_{metric_name}.png"
    plot_save_path = os.path.join(output_dir, plot_filename)
    
    try:
        plt.savefig(plot_save_path)
        print(f"Saved plot: {plot_save_path}")
    except Exception as e:
        print(f"Error saving plot {plot_save_path}: {e}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compare final test set performance of different trained models.")
    parser.add_argument("--model-dirs", type=str, nargs='+', required=True)
    parser.add_argument("--output-plot-dir", type=str, default="./plots")

    args = parser.parse_args()

    os.makedirs(args.output_plot_dir, exist_ok=True)

    all_performance_data = []
    model_display_names = []

    for model_dir in args.model_dirs:
        performance_file = os.path.join(model_dir, "training_logs", "final_test_set_performance.json")
        if os.path.exists(performance_file):
            try:
                with open(performance_file, 'r') as f:
                    data = json.load(f)
                    all_performance_data.append(data)
                    display_name = data.get('model_name', os.path.basename(model_dir))
                    model_display_names.append(display_name)
                print(f"Loaded performance data from: {performance_file}")
            except Exception as e:
                print(f"Error loading or parsing {performance_file}: {e}")
        else:
            print(f"Warning: Performance file not found in {model_dir} (expected at {performance_file}). Skipping.")

    if not all_performance_data:
        print("Error: No performance data loaded. Exiting.")
        return
    
    metrics = ['test_total_loss', 'train_total_loss']

    # Plot each specified metric
    for metric_key in metrics:
        metric_values = []
        current_model_names_for_metric = []
        for i, data in enumerate(all_performance_data):
            if metric_key in data:
                metric_values.append(data[metric_key])
                current_model_names_for_metric.append(model_display_names[i]) 
            else:
                print(f"Warning: Metric '{metric_key}' not found for model in {args.model_dirs[i]}. Skipping this model for this metric.")
        
        if metric_values: 
            plot_metric_comparison(current_model_names_for_metric, metric_values, metric_key, args.output_plot_dir)
        else:
            print(f"No data found for any model for metric: {metric_key}. Skipping plot.")

if __name__ == '__main__':
    main() 