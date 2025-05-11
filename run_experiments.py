import subprocess
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run multiple training experiments for PoseNet.")
    parser.add_argument("--base-output-dir", type=str, default="../experiment_results")
    args = parser.parse_args()

    # Training parameters
    annotation_file = '../speed/speed/train.json'
    img_dir = '../speed/speed/images/train/'
    val_split_ratio = 0.2
    test_annotation_file = '../speed/speed/lightbox/test.json'
    test_img_dir = '../speed/speed/lightbox/images/'
    lr = 1e-4
    beta_loss = 50.0
    batch_size = 64
    num_epochs = 10
    num_workers = 8

    os.makedirs(args.base_output_dir, exist_ok=True)

    experiments = [
        {"arch": "efficientnet_v2_s", "freeze": False, "name_suffix": "EffNet_NoFreeze"},
        {"arch": "efficientnet_v2_s", "freeze": True,  "name_suffix": "EffNet_Freeze"},
        {"arch": "resnet34",          "freeze": False, "name_suffix": "ResNet34_NoFreeze"},
        {"arch": "resnet34",          "freeze": True,  "name_suffix": "ResNet34_Freeze"},
    ]
    for i, exp_config in enumerate(experiments):
        model_name = f"{exp_config['name_suffix']}"
        freeze_flag_str = exp_config["freeze"]

        command = [
            "python", "train.py",
            "--architecture", exp_config["arch"],
            "--freeze-early-backbone-layers", freeze_flag_str,
            "--model-name", model_name,
            "--output-model-path", args.base_output_dir,
            "--annotation-file", annotation_file,
            "--img-dir", img_dir,
            "--val-split-ratio", str(val_split_ratio),
            "--test-annotation-file", test_annotation_file,
            "--test-img-dir", test_img_dir,
            "--lr", str(lr),
            "--beta-loss", str(beta_loss),
            "--batch-size", str(batch_size),
            "--num-epochs", str(num_epochs),
            "--num-workers", str(num_workers)
        ]

        print(f"\n--- Starting Experiment {i+1}/{len(experiments)}: {model_name} ---")
        print(f"Command: {' '.join(command)}")
        
        try:
            subprocess.run(command, check=True)
            print(f"--- Finished Experiment {i+1}/{len(experiments)}: {model_name} ---")
        except Exception as e:
            print(f"Error during experiment {model_name}: {e}")
            print(f"--- Experiment {i+1}/{len(experiments)}: {model_name} FAILED ---")

if __name__ == "__main__":
    main() 