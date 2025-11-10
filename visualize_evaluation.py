"""Visualize evaluation results for milestone report.

Creates publication-quality comparison plots for static vs hindered evaluation.
"""
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_results(csv_path: Path) -> Tuple[List[Dict], float]:
    """Load evaluation results from CSV file.
    
    Returns:
        results: List of episode results
        success_rate: Overall success rate
    """
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'episode': int(row['episode']),
                'success': float(row['success']),
                'reward': float(row['reward'])
            })
    
    if results:
        success_rate = sum(r['success'] for r in results) / len(results)
    else:
        success_rate = 0.0
    
    return results, success_rate


def create_comparison_plots(
    static_results: List[Dict],
    hindered_results: List[Dict],
    static_rate: float,
    hindered_rate: float,
    output_dir: Path
) -> None:
    """Create comprehensive comparison visualizations."""
    
    # Set style for publication-quality figures
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'serif'
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(14, 10))
    
    # 1. Success Rate Comparison (Bar Chart)
    ax1 = plt.subplot(2, 3, 1)
    modes = ['Static', 'Hindered']
    rates = [static_rate * 100, hindered_rate * 100]
    colors = ['#2ecc71', '#e74c3c']
    
    bars = ax1.bar(modes, rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Success Rate (%)', fontweight='bold')
    ax1.set_title('Overall Success Rate Comparison', fontweight='bold', fontsize=12)
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 2. Episode-by-Episode Success (Line Plot)
    ax2 = plt.subplot(2, 3, 2)
    static_successes = [r['success'] for r in static_results]
    hindered_successes = [r['success'] for r in hindered_results]
    
    ax2.plot(range(1, len(static_successes) + 1), static_successes, 
             'o-', color='#2ecc71', label='Static', linewidth=2, markersize=6, alpha=0.7)
    ax2.plot(range(1, len(hindered_successes) + 1), hindered_successes,
             's-', color='#e74c3c', label='Hindered', linewidth=2, markersize=6, alpha=0.7)
    
    ax2.set_xlabel('Episode', fontweight='bold')
    ax2.set_ylabel('Success (1=Success, 0=Fail)', fontweight='bold')
    ax2.set_title('Episode-by-Episode Success', fontweight='bold', fontsize=12)
    ax2.set_ylim([-0.1, 1.1])
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(alpha=0.3)
    
    # 3. Cumulative Success Rate
    ax3 = plt.subplot(2, 3, 3)
    static_cumsum = np.cumsum(static_successes) / np.arange(1, len(static_successes) + 1)
    hindered_cumsum = np.cumsum(hindered_successes) / np.arange(1, len(hindered_successes) + 1)
    
    ax3.plot(range(1, len(static_cumsum) + 1), static_cumsum * 100,
             '-', color='#2ecc71', label='Static', linewidth=2.5)
    ax3.plot(range(1, len(hindered_cumsum) + 1), hindered_cumsum * 100,
             '-', color='#e74c3c', label='Hindered', linewidth=2.5)
    
    ax3.set_xlabel('Episode', fontweight='bold')
    ax3.set_ylabel('Cumulative Success Rate (%)', fontweight='bold')
    ax3.set_title('Cumulative Success Rate Over Time', fontweight='bold', fontsize=12)
    ax3.legend(loc='best', framealpha=0.9)
    ax3.grid(alpha=0.3)
    
    # 4. Reward Distribution (Violin Plot)
    ax4 = plt.subplot(2, 3, 4)
    static_rewards = [r['reward'] for r in static_results]
    hindered_rewards = [r['reward'] for r in hindered_results]
    
    data = [static_rewards, hindered_rewards]
    parts = ax4.violinplot(data, positions=[0, 1], showmeans=True, showmedians=True)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(modes)
    ax4.set_ylabel('Total Reward', fontweight='bold')
    ax4.set_title('Reward Distribution', fontweight='bold', fontsize=12)
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Success/Failure Count (Stacked Bar)
    ax5 = plt.subplot(2, 3, 5)
    static_success_count = sum(static_successes)
    static_fail_count = len(static_successes) - static_success_count
    hindered_success_count = sum(hindered_successes)
    hindered_fail_count = len(hindered_successes) - hindered_success_count
    
    x = np.arange(2)
    width = 0.5
    
    p1 = ax5.bar(x, [static_success_count, hindered_success_count], width, 
                 label='Success', color='#2ecc71', alpha=0.8)
    p2 = ax5.bar(x, [static_fail_count, hindered_fail_count], width,
                 bottom=[static_success_count, hindered_success_count],
                 label='Failure', color='#e74c3c', alpha=0.8)
    
    ax5.set_ylabel('Episode Count', fontweight='bold')
    ax5.set_title('Success vs Failure Count', fontweight='bold', fontsize=12)
    ax5.set_xticks(x)
    ax5.set_xticklabels(modes)
    ax5.legend(loc='best', framealpha=0.9)
    ax5.grid(axis='y', alpha=0.3)
    
    # Add counts on bars
    for i, (succ, fail) in enumerate([(static_success_count, static_fail_count),
                                       (hindered_success_count, hindered_fail_count)]):
        ax5.text(i, succ/2, str(int(succ)), ha='center', va='center', 
                fontweight='bold', color='white', fontsize=11)
        ax5.text(i, succ + fail/2, str(int(fail)), ha='center', va='center',
                fontweight='bold', color='white', fontsize=11)
    
    # 6. Statistics Summary (Text)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    static_avg_reward = np.mean(static_rewards)
    hindered_avg_reward = np.mean(hindered_rewards)
    static_std_reward = np.std(static_rewards)
    hindered_std_reward = np.std(hindered_rewards)
    
    summary_text = f"""
    EVALUATION SUMMARY
    
    Static Environment:
      • Success Rate: {static_rate*100:.1f}%
      • Episodes: {len(static_results)}
      • Avg Reward: {static_avg_reward:.2f} ± {static_std_reward:.2f}
    
    Hindered Environment:
      • Success Rate: {hindered_rate*100:.1f}%
      • Episodes: {len(hindered_results)}
      • Avg Reward: {hindered_avg_reward:.2f} ± {hindered_avg_reward:.2f}
    
    Performance Gap:
      • Δ Success: {(static_rate - hindered_rate)*100:.1f}%
      • Δ Reward: {static_avg_reward - hindered_avg_reward:.2f}
    
    Model: VLA with DINOv2 + BERT
    Training: 9 episodes, 20 epochs
    Features: Temporal awareness (timestep)
    """
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'evaluation_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved comprehensive comparison plot: {output_path}")
    
    # Also save as PDF for LaTeX reports
    output_path_pdf = output_dir / 'evaluation_comparison.pdf'
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✅ Saved PDF version: {output_path_pdf}")
    
    plt.close()


def create_simple_bar_chart(
    static_rate: float,
    hindered_rate: float,
    output_dir: Path
) -> None:
    """Create a simple, clean bar chart for presentations."""
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    modes = ['Static', 'Hindered']
    rates = [static_rate * 100, hindered_rate * 100]
    colors = ['#3498db', '#e67e22']
    
    bars = ax.bar(modes, rates, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    ax.set_ylabel('Success Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Policy Performance: Static vs Hindered Environment', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{rate:.1f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    
    output_path = output_dir / 'simple_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved simple bar chart: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize evaluation results")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Directory containing static_eval.csv and hindered_eval.csv")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Directory to save plots")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    static_csv = results_dir / "static_eval.csv"
    hindered_csv = results_dir / "hindered_eval.csv"
    
    if not static_csv.exists():
        print(f"❌ Error: {static_csv} not found!")
        print("   Run evaluation first:")
        print("   python evaluate_bc_mujoco.py --checkpoint runs/dinov2_bc_best.pt --episodes-static 50 --episodes-hindered 50")
        return
    
    if not hindered_csv.exists():
        print(f"❌ Error: {hindered_csv} not found!")
        print("   Run hindered evaluation first:")
        print("   python evaluate_bc_mujoco.py --checkpoint runs/dinov2_bc_best.pt --episodes-static 50 --episodes-hindered 50")
        return
    
    print("Loading evaluation results...")
    static_results, static_rate = load_results(static_csv)
    hindered_results, hindered_rate = load_results(hindered_csv)
    
    print(f"\n📊 Results Summary:")
    print(f"  Static:   {static_rate*100:.1f}% success ({len(static_results)} episodes)")
    print(f"  Hindered: {hindered_rate*100:.1f}% success ({len(hindered_results)} episodes)")
    print(f"  Gap:      {(static_rate - hindered_rate)*100:.1f}%")
    
    print("\n🎨 Creating visualizations...")
    
    # Create comprehensive comparison plot
    create_comparison_plots(
        static_results, hindered_results,
        static_rate, hindered_rate,
        output_dir
    )
    
    # Create simple bar chart
    create_simple_bar_chart(static_rate, hindered_rate, output_dir)
    
    print(f"\n✅ All plots saved to: {output_dir}/")
    print("\nPlots created:")
    print("  1. evaluation_comparison.png - Comprehensive 6-panel comparison")
    print("  2. evaluation_comparison.pdf - PDF version for LaTeX reports")
    print("  3. simple_comparison.png     - Clean bar chart for presentations")


if __name__ == "__main__":
    main()

