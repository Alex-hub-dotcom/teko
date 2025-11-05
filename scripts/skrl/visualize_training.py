#!/usr/bin/env python3
"""
Training Results Visualization
==============================
Analyze and visualize training results from TensorBoard logs.
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

def load_tensorboard_data(log_dir):
    """Load data from TensorBoard event files."""
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    data = {}
    
    # Load scalar data
    tags = ea.Tags()['scalars']
    for tag in tags:
        events = ea.Scalars(tag)
        data[tag] = {
            'steps': [e.step for e in events],
            'values': [e.value for e in events]
        }
    
    return data

def plot_training_curves(log_dir, output_dir=None):
    """Generate training curve plots."""
    print(f"Loading data from {log_dir}...")
    
    try:
        data = load_tensorboard_data(log_dir)
    except Exception as e:
        print(f"Error loading TensorBoard data: {e}")
        return
    
    if output_dir is None:
        output_dir = log_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('TEKO Vision-Based Docking Training Results', fontsize=16)
    
    # Plot 1: Episode Reward
    if 'Episode/Reward' in data:
        ax = axes[0, 0]
        steps = data['Episode/Reward']['steps']
        values = data['Episode/Reward']['values']
        
        # Plot raw rewards
        ax.plot(steps, values, alpha=0.3, label='Raw')
        
        # Plot moving average
        if len(values) > 50:
            window = min(50, len(values) // 10)
            moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
            ax.plot(steps[window-1:], moving_avg, linewidth=2, label=f'MA({window})')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Success Rate
    if 'Episode/Success_Rate' in data:
        ax = axes[0, 1]
        steps = data['Episode/Success_Rate']['steps']
        values = data['Episode/Success_Rate']['values']
        
        ax.plot(steps, values, linewidth=2, color='green')
        ax.axhline(y=0.7, color='r', linestyle='--', label='Curriculum Threshold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate (100-episode moving average)')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Episode Length
    if 'Episode/Length' in data:
        ax = axes[1, 0]
        steps = data['Episode/Length']['steps']
        values = data['Episode/Length']['values']
        
        ax.plot(steps, values, alpha=0.3, label='Raw')
        
        # Moving average
        if len(values) > 50:
            window = min(50, len(values) // 10)
            moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
            ax.plot(steps[window-1:], moving_avg, linewidth=2, label=f'MA({window})')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Length')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Reward Components
    ax = axes[1, 1]
    reward_components = [
        'Reward_Components/distance',
        'Reward_Components/alignment',
        'Reward_Components/orientation',
        'Reward_Components/velocity_penalty',
        'Reward_Components/oscillation_penalty'
    ]
    
    for component in reward_components:
        if component in data:
            steps = data[component]['steps']
            values = data[component]['values']
            
            # Moving average
            if len(values) > 20:
                window = 20
                moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
                label = component.split('/')[-1]
                ax.plot(steps[window-1:], moving_avg, label=label, linewidth=2)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Value')
    ax.set_title('Reward Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    
    plt.show()

def print_summary(log_dir):
    """Print training summary statistics."""
    config_path = os.path.join(log_dir, 'config.json')
    summary_path = os.path.join(log_dir, 'training_summary.txt')
    
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    # Load config
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"\nConfiguration:")
        print(f"  Timestamp: {config.get('timestamp', 'N/A')}")
        print(f"  Num Envs: {config.get('num_envs', 'N/A')}")
        print(f"  Total Timesteps: {config.get('total_timesteps', 'N/A')}")
        print(f"  Curriculum: {config.get('curriculum', 'N/A')}")
    
    # Load summary
    if os.path.exists(summary_path):
        print(f"\n{open(summary_path).read()}")
    
    # Load TensorBoard data for final statistics
    try:
        data = load_tensorboard_data(log_dir)
        
        if 'Episode/Reward' in data:
            rewards = data['Episode/Reward']['values']
            print(f"\nReward Statistics (last 100 episodes):")
            print(f"  Mean: {np.mean(rewards[-100:]):.2f}")
            print(f"  Std:  {np.std(rewards[-100:]):.2f}")
            print(f"  Max:  {np.max(rewards[-100:]):.2f}")
            print(f"  Min:  {np.min(rewards[-100:]):.2f}")
        
        if 'Episode/Success_Rate' in data:
            success_rate = data['Episode/Success_Rate']['values'][-1]
            print(f"\nFinal Success Rate: {success_rate:.2%}")
        
        if 'Episode/Length' in data:
            lengths = data['Episode/Length']['values']
            print(f"\nEpisode Length (last 100):")
            print(f"  Mean: {np.mean(lengths[-100:]):.1f} steps")
        
    except Exception as e:
        print(f"\nCould not load TensorBoard data: {e}")
    
    print("\n" + "="*70 + "\n")

def compare_runs(log_dirs):
    """Compare multiple training runs."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Training Runs Comparison', fontsize=16)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(log_dirs)))
    
    for idx, log_dir in enumerate(log_dirs):
        try:
            data = load_tensorboard_data(log_dir)
            run_name = os.path.basename(log_dir)
            
            # Plot rewards
            if 'Episode/Reward' in data:
                ax = axes[0]
                steps = data['Episode/Reward']['steps']
                values = data['Episode/Reward']['values']
                
                window = min(50, len(values) // 10)
                if len(values) > window:
                    moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
                    ax.plot(steps[window-1:], moving_avg, label=run_name, 
                           color=colors[idx], linewidth=2)
            
            # Plot success rate
            if 'Episode/Success_Rate' in data:
                ax = axes[1]
                steps = data['Episode/Success_Rate']['steps']
                values = data['Episode/Success_Rate']['values']
                ax.plot(steps, values, label=run_name, color=colors[idx], linewidth=2)
        
        except Exception as e:
            print(f"Error loading {log_dir}: {e}")
    
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward (MA)')
    axes[0].set_title('Episode Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Success Rate')
    axes[1].set_title('Success Rate')
    axes[1].set_ylim([0, 1])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('runs_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved comparison to runs_comparison.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize TEKO training results')
    parser.add_argument('log_dir', type=str, nargs='?', 
                       default='/workspace/teko/runs',
                       help='Path to TensorBoard log directory')
    parser.add_argument('--compare', nargs='+', 
                       help='Compare multiple runs (provide multiple log dirs)')
    parser.add_argument('--latest', action='store_true',
                       help='Use the latest run in log_dir')
    
    args = parser.parse_args()
    
    if args.compare:
        print("Comparing multiple runs...")
        compare_runs(args.compare)
    else:
        # Find log directory
        if args.latest:
            # Find latest subdirectory
            runs = sorted(Path(args.log_dir).glob('teko_ppo_*'))
            if not runs:
                print(f"No runs found in {args.log_dir}")
                return
            log_dir = str(runs[-1])
            print(f"Using latest run: {log_dir}")
        else:
            log_dir = args.log_dir
        
        if not os.path.exists(log_dir):
            print(f"Error: {log_dir} does not exist")
            return
        
        # Generate plots and summary
        print_summary(log_dir)
        plot_training_curves(log_dir)

if __name__ == '__main__':
    main()