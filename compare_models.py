import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

def plot_train_test_loss_comparison(model_names, train_loss_values, test_loss_values, output_dir):
    """
    Generates and saves a bar plot comparing train and test loss side-by-side for models.
    """
    plt.figure(figsize=(10 + len(model_names) * 0.8, 7))
    
    num_models = len(model_names)
    bar_positions = np.arange(num_models)
    bar_width = 0.35

    bars1 = plt.bar(bar_positions - bar_width/2, train_loss_values, bar_width, label='Train Loss', color='skyblue')
    bars2 = plt.bar(bar_positions + bar_width/2, test_loss_values, bar_width, label='Test Loss', color='coral')

    plt.ylabel('Loss Value')
    plt.title('Comparison of Model Performance: Train vs Test Loss')
    plt.xticks(bar_positions, model_names, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    max_combined_loss = 0
    if train_loss_values and test_loss_values:
        max_combined_loss = max(max(train_loss_values, default=0), max(test_loss_values, default=0))

    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * max_combined_loss, f'{yval:.4f}', ha='center', va='bottom')

    plot_filename = "comparison_train_vs_test_loss.png"
    plot_save_path = os.path.join(output_dir, plot_filename)
    
    plt.savefig(plot_save_path)
    print(f"Saved plot: {plot_save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compare final test set performance of different trained models.")
    parser.add_argument("--model-dirs", type=str, nargs='+', required=True)
    parser.add_argument("--output-plot-dir", type=str, default="./plots")

    args = parser.parse_args()

    os.makedirs(args.output_plot_dir, exist_ok=True)

    model_names_for_plot = []
    train_loss_values = []
    test_loss_values = []

    for model_dir in args.model_dirs:
        performance_file = os.path.join(model_dir, "training_logs", "final_test_set_performance.json")
        with open(performance_file, 'r') as f:
            data = json.load(f)
        
        model_name = data.get('model_name', os.path.basename(model_dir))
        
        model_names_for_plot.append(model_name)
        train_loss_values.append(data['best_val_model_train_loss'])
        test_loss_values.append(data['test_total_loss'])
        print(f"Loaded performance data for {model_name} from: {performance_file}")

    plot_train_test_loss_comparison(model_names_for_plot, train_loss_values, test_loss_values, args.output_plot_dir)

if __name__ == '__main__':
    main() 