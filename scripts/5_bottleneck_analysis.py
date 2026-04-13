"""
Step 5: Bottleneck analysis - find struggle points in recipes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from utils import load_hd_epic_data

def analyze_bottlenecks(G, verb_classes, noun_classes, top_n=15):
    """
    Find transitions with longest pauses (struggle points)
    """
    
    print("\n" + "="*80)
    print("BOTTLENECK ANALYSIS")
    print("="*80)
    
    # Extract all edges
    bottlenecks = []
    
    for u, v, key, data in G.edges(data=True, keys=True):
        bottlenecks.append({
            'from_action': u,
            'to_action': v,
            'frequency': data['weight'],
            'avg_pause': data['avg_pause'],
            'max_pause': data['max_pause'],
            'std_pause': data['std_pause']
        })
    
    bottlenecks_df = pd.DataFrame(bottlenecks)
    
    # Sort by average pause
    bottlenecks_df = bottlenecks_df.sort_values('avg_pause', ascending=False)
    
    print(f"\nTop {top_n} Bottlenecks (Longest Pauses):")
    print("="*100)
    print(f"{'Rank':<6} {'From Action':<30} {'To Action':<30} {'Freq':<6} {'Avg Pause':<12} {'Max Pause':<12}")
    print("="*100)
    
    for i, row in bottlenecks_df.head(top_n).iterrows():
        from_short = row['from_action'][:28] + '..' if len(row['from_action']) > 30 else row['from_action']
        to_short = row['to_action'][:28] + '..' if len(row['to_action']) > 30 else row['to_action']
        
        print(f"{i+1:<6} {from_short:<30} {to_short:<30} "
              f"{row['frequency']:<6} {row['avg_pause']:<12.1f} {row['max_pause']:<12.1f}")
    
    return bottlenecks_df


def visualize_bottlenecks(bottlenecks_df, top_n=15):
    """
    Visualize top bottlenecks
    """
    
    print("\n" + "="*80)
    print("VISUALIZING BOTTLENECKS")
    print("="*80)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Top bottlenecks by average pause
    top = bottlenecks_df.head(top_n)
    
    # Create labels
    labels = [f"{row['from_action'][:15]}... → {row['to_action'][:15]}..." 
             for _, row in top.iterrows()]
    
    colors = ['red' if p > 30 else 'orange' if p > 10 else 'yellow' 
             for p in top['avg_pause']]
    
    ax1.barh(range(len(top)), top['avg_pause'], color=colors, edgecolor='black')
    ax1.set_yticks(range(len(top)))
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel('Average Pause (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Top {top_n} Bottlenecks by Pause Duration', 
                 fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Add threshold lines
    ax1.axvline(x=30, color='red', linestyle='--', linewidth=2, 
               label='High struggle (30s)')
    ax1.axvline(x=10, color='orange', linestyle='--', linewidth=2,
               label='Medium struggle (10s)')
    ax1.legend(fontsize=10)
    
    # Plot 2: Frequency vs Pause scatter
    ax2.scatter(bottlenecks_df['frequency'], bottlenecks_df['avg_pause'],
               s=bottlenecks_df['max_pause']*2,
               c=bottlenecks_df['avg_pause'],
               cmap='YlOrRd',
               alpha=0.6,
               edgecolors='black',
               linewidth=0.5)
    
    ax2.set_xlabel('Transition Frequency', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Pause (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Bottleneck Frequency vs. Struggle Intensity',
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add quadrant lines
    median_freq = bottlenecks_df['frequency'].median()
    median_pause = bottlenecks_df['avg_pause'].median()
    
    ax2.axhline(y=median_pause, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=median_freq, color='gray', linestyle='--', alpha=0.5)
    
    # Annotate quadrants
    ax2.text(0.95, 0.95, 'High Impact\n(Frequent + Slow)',
            transform=ax2.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.3),
            fontsize=10, fontweight='bold')
    
    ax2.text(0.05, 0.95, 'Rare Struggles',
            transform=ax2.transAxes, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3),
            fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../outputs/figures/bottleneck_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Bottleneck visualization saved")
    plt.close()


def create_intervention_plan(bottlenecks_df, top_n=10):
    """
    Create robot intervention plan based on bottlenecks
    """
    
    print("\n" + "="*80)
    print("ROBOT INTERVENTION PLAN")
    print("="*80)
    
    interventions = []
    
    for i, row in bottlenecks_df.head(top_n).iterrows():
        # Determine intervention type
        if row['avg_pause'] > 30:
            intervention_type = 'proactive_guidance'
            urgency = 'high'
            robot_action = f"After user completes {row['from_action']}, " \
                          f"immediately suggest: 'The next step is {row['to_action']}. Would you like help?'"
        elif row['avg_pause'] > 10:
            intervention_type = 'reminder'
            urgency = 'medium'
            robot_action = f"If pause exceeds 15s after {row['from_action']}, " \
                          f"remind user: 'Next step is {row['to_action']}.'"
        else:
            intervention_type = 'monitoring'
            urgency = 'low'
            robot_action = f"Monitor transition from {row['from_action']} to {row['to_action']}"
        
        interventions.append({
            'rank': i + 1,
            'from_action': row['from_action'],
            'to_action': row['to_action'],
            'avg_pause': row['avg_pause'],
            'frequency': row['frequency'],
            'intervention_type': intervention_type,
            'urgency': urgency,
            'robot_action': robot_action
        })
    
    intervention_df = pd.DataFrame(interventions)
    
    # Save
    intervention_df.to_csv('../outputs/tables/robot_intervention_plan.csv', index=False)
    print("\n✓ Intervention plan saved to ../outputs/tables/robot_intervention_plan.csv")
    
    # Display
    print("\nTop 5 Intervention Priorities:")
    for _, row in intervention_df.head(5).iterrows():
        print(f"\n{row['rank']}. Priority: {row['urgency'].upper()}")
        print(f"   Transition: {row['from_action'][:40]}... → {row['to_action'][:40]}...")
        print(f"   Avg pause: {row['avg_pause']:.1f}s (occurs {row['frequency']} times)")
        print(f"   Robot action: {row['robot_action'][:100]}...")
    
    return intervention_df


def main():
    # Load graph
    with open('../outputs/graphs/motion_graph.pkl', 'rb') as f:
        G = pickle.load(f)
    
    data = load_hd_epic_data('..')
    
    # Analyze bottlenecks
    bottlenecks_df = analyze_bottlenecks(G, data['verb_classes'], data['noun_classes'])
    
    # Save
    bottlenecks_df.to_csv('../outputs/tables/bottlenecks.csv', index=False)
    print("\n✓ Bottlenecks saved to ../outputs/tables/bottlenecks.csv")
    
    # Visualize
    visualize_bottlenecks(bottlenecks_df)
    
    # Create intervention plan
    create_intervention_plan(bottlenecks_df)
    
    print("\n" + "="*80)
    print("BOTTLENECK ANALYSIS COMPLETE")
    print("="*80)
    print("\nAll analysis files are in ../outputs/")
    print("  - Figures: ../outputs/figures/")
    print("  - Tables: ../outputs/tables/")
    print("  - Graphs: ../outputs/graphs/")

if __name__ == "__main__":
    main()