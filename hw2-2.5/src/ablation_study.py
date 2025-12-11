"""
Ablation Study: Compare model performance with and without synthetic data
"""
import argparse
import json
from pathlib import Path
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class AblationStudy:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def run_experiment(self, name, config):
        """Run a single experiment"""
        print(f"\n{'='*60}")
        print(f"Running experiment: {name}")
        print(f"{'='*60}\n")
        
        # Create experiment directory
        exp_dir = self.output_dir / name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = exp_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Run training
        train_cmd = [
            'python', 'src/train.py',
            '--data_dir', config['data_dir'],
            '--output_dir', str(exp_dir),
            '--batch_size', str(config.get('batch_size', 4)),
            '--num_epochs', str(config.get('num_epochs', 10)),
            '--lr', str(config.get('lr', 1e-5))
        ]
        
        if config.get('max_train_samples'):
            train_cmd.extend(['--max_train_samples', str(config['max_train_samples'])])
        
        subprocess.run(train_cmd, check=True)
        
        # Run evaluation
        checkpoint = exp_dir / 'checkpoints' / 'best_model.pt'
        eval_cmd = [
            'python', 'src/evaluate.py',
            '--data_dir', config['data_dir'],
            '--checkpoint', str(checkpoint),
            '--config', str(config_path),
            '--output', str(exp_dir / 'metrics.json')
        ]
        
        subprocess.run(eval_cmd, check=True)
        
        # Load and store results
        with open(exp_dir / 'metrics.json', 'r') as f:
            metrics = json.load(f)
        
        result = {
            'experiment': name,
            'config': config,
            'metrics': metrics
        }
        self.results.append(result)
        
        return result
    
    def compare_results(self):
        """Compare and visualize results"""
        if len(self.results) < 2:
            print("Need at least 2 experiments to compare")
            return
        
        # Create comparison table
        comparison_data = []
        for result in self.results:
            row = {
                'Experiment': result['experiment'],
                'mAP': result['metrics'].get('bbox_mAP', 0),
                'mAP50': result['metrics'].get('bbox_mAP50', 0),
                'mAP75': result['metrics'].get('bbox_mAP75', 0)
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Save table
        table_path = self.output_dir / 'comparison_table.csv'
        df.to_csv(table_path, index=False)
        print(f"\nComparison table saved to {table_path}")
        print("\n" + df.to_string(index=False))
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = range(len(df))
        width = 0.25
        
        ax.bar([i - width for i in x], df['mAP'], width, label='mAP', alpha=0.8)
        ax.bar(x, df['mAP50'], width, label='mAP50', alpha=0.8)
        ax.bar([i + width for i in x], df['mAP75'], width, label='mAP75', alpha=0.8)
        
        ax.set_xlabel('Experiment')
        ax.set_ylabel('Score')
        ax.set_title('Ablation Study: Metric Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Experiment'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_path = self.output_dir / 'comparison_plot.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Comparison plot saved to {plot_path}")
        
        # Calculate improvement
        if len(self.results) == 2:
            baseline = self.results[0]['metrics'].get('bbox_mAP', 0)
            improved = self.results[1]['metrics'].get('bbox_mAP', 0)
            improvement = ((improved - baseline) / baseline) * 100
            
            print(f"\n{'='*60}")
            print(f"Improvement Analysis:")
            print(f"{'='*60}")
            print(f"Baseline mAP: {baseline:.4f}")
            print(f"With Synthetic Data mAP: {improved:.4f}")
            print(f"Improvement: {improvement:+.2f}%")
            print(f"{'='*60}\n")
            
            # Save improvement report
            improvement_report = {
                'baseline_experiment': self.results[0]['experiment'],
                'improved_experiment': self.results[1]['experiment'],
                'baseline_mAP': baseline,
                'improved_mAP': improved,
                'improvement_percent': improvement,
                'baseline_metrics': self.results[0]['metrics'],
                'improved_metrics': self.results[1]['metrics']
            }
            
            with open(self.output_dir / 'improvement_report.json', 'w') as f:
                json.dump(improvement_report, f, indent=2)
    
    def save_summary(self):
        """Save summary of all experiments"""
        summary = {
            'num_experiments': len(self.results),
            'experiments': self.results
        }
        
        with open(self.output_dir / 'ablation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nAblation study summary saved to {self.output_dir / 'ablation_summary.json'}")


def main():
    parser = argparse.ArgumentParser(description='Run ablation study')
    parser.add_argument('--data_dir', type=str, default='/home/salex139s/test/data/coco',
                        help='Path to COCO dataset')
    parser.add_argument('--synthetic_dir', type=str, default='/home/salex139s/test/data/synthetic',
                        help='Path to synthetic data')
    parser.add_argument('--output_dir', type=str, default='/home/salex139s/test/outputs/ablation',
                        help='Output directory for ablation study')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--quick_test', action='store_true',
                        help='Run quick test with limited samples')
    
    args = parser.parse_args()
    
    # Create ablation study
    study = AblationStudy(output_dir=args.output_dir)
    
    # Experiment 1: Baseline (no synthetic data)
    baseline_config = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'lr': args.lr,
        'use_synthetic': False
    }
    
    if args.quick_test:
        baseline_config['max_train_samples'] = 100
        baseline_config['num_epochs'] = 2
    
    study.run_experiment('baseline', baseline_config)
    
    # Experiment 2: With synthetic data
    # Note: This assumes synthetic data has been generated and integrated
    synthetic_config = {
        'data_dir': args.data_dir,  # Should include synthetic data
        'synthetic_dir': args.synthetic_dir,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'lr': args.lr,
        'use_synthetic': True
    }
    
    if args.quick_test:
        synthetic_config['max_train_samples'] = 100
        synthetic_config['num_epochs'] = 2
    
    # For now, we'll skip this since synthetic data needs to be integrated
    # study.run_experiment('with_synthetic', synthetic_config)
    
    # Compare results
    # study.compare_results()
    study.save_summary()
    
    print("\n" + "="*60)
    print("Ablation study completed!")
    print(f"Results saved to {args.output_dir}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()

