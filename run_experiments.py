import json
import sys
import os

from train_experiments import *

def run_experiment(exp_config):
    name = exp_config['name']
    print(f"\n{'='*50}")
    print(f"Running experiment: {name}")
    print(f"{'='*50}")
    
    globals().update({k: v for k, v in exp_config.items() if k != 'name'})
    
    global train_loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory)
    
    train_model(
        use_lower_boundary=exp_config['use_lower_boundary'],
        use_upper_boundary=exp_config['use_upper_boundary'],
        model_save_name=f"model_{name}.pt"
    )
    
    print(f"Completed experiment: {name}")

def main():
    with open('configs.json', 'r') as f:
        config = json.load(f)
    
    experiments = config['experiments']
    print(f"Found {len(experiments)} experiments to run")
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Starting: {exp['name']}")
        try:
            run_experiment(exp)
        except Exception as e:
            print(f"ERROR in experiment {exp['name']}: {e}")
            continue
    
    print(f"\n{'='*50}")
    print("All experiments completed!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()