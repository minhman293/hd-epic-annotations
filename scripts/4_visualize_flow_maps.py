"""
Step 4: Create flow map visualizations comparing normal vs abnormal sessions
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
import pickle
from pathlib import Path
from utils import (load_hd_epic_data, get_action_name, count_loops, calculate_pause)


def load_selected_recipe_files(outputs_dir='../outputs'):
    """Load latest recipe-specific selection files from outputs directory."""
    outputs_path = Path(outputs_dir)

    recipe_json_candidates = sorted(
        outputs_path.glob('selected_recipe_*.json'),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not recipe_json_candidates:
        raise FileNotFoundError(
            "No selection file found. Run 2_recipe_selector.py first "
            "to create selected_recipe_<recipe_id>.json"
        )

    recipe_json_path = recipe_json_candidates[0]
    with open(recipe_json_path, 'r') as f:
        recipe_info = json.load(f)

    recipe_id = recipe_info['recipe_id']
    recipe_narrations_path = outputs_path / f'recipe_narrations_{recipe_id}.pkl'
    if not recipe_narrations_path.exists():
        raise FileNotFoundError(
            f"Missing narration file: {recipe_narrations_path}. "
            "Run 2_recipe_selector.py again to regenerate outputs."
        )

    recipe_narrations = pd.read_pickle(recipe_narrations_path)
    return recipe_info, recipe_narrations

def categorize_sessions(recipe_narrations, video_ids):
    """
    Categorize sessions as normal or abnormal based on:
    - Number of oscillation loops
    - Total session duration
    - Average pause duration
    """
    
    print("\n" + "="*80)
    print("CATEGORIZING SESSIONS")
    print("="*80)
    
    session_stats = []
    
    for video_id in video_ids:
        video_actions = recipe_narrations[recipe_narrations['video_id'] == video_id].copy()
        
        if len(video_actions) < 3:
            continue
        
        # Get main actions
        main_actions = []
        for idx, row in video_actions.iterrows():
            if row['main_action_classes'] and len(row['main_action_classes']) > 0:
                main_actions.append(row['main_action_classes'][0])
        
        # Count loops
        loop_count = count_loops(pd.Series([str(a) for a in main_actions]))
        
        # Calculate duration
        total_duration = video_actions['end_timestamp'].max() - video_actions['start_timestamp'].min()
        
        # Calculate pauses
        video_actions['pause_after'] = calculate_pause(video_actions)
        avg_pause = video_actions['pause_after'].mean()
        
        # Categorize
        anomaly_score = 0
        if loop_count > 5:
            anomaly_score += 2
        if total_duration > video_actions['end_timestamp'].median() * 1.5:
            anomaly_score += 1
        if avg_pause > 15:
            anomaly_score += 1
        
        is_abnormal = anomaly_score >= 2
        
        session_stats.append({
            'video_id': video_id,
            'loops': loop_count,
            'duration': total_duration,
            'avg_pause': avg_pause,
            'anomaly_score': anomaly_score,
            'is_abnormal': is_abnormal
        })
        
        print(f"{video_id}: loops={loop_count}, duration={total_duration:.1f}s, "
              f"avg_pause={avg_pause:.1f}s, abnormal={is_abnormal}")
    
    return pd.DataFrame(session_stats)


def create_flow_map_comparison(recipe_narrations, session_stats, verb_classes, noun_classes):
    """
    Create flow maps comparing normal vs abnormal sessions
    Like Figure 11 in the example
    """
    
    print("\n" + "="*80)
    print("CREATING FLOW MAP COMPARISON")
    print("="*80)
    
    # Select 2 normal and 2 abnormal sessions
    normal_sessions = session_stats[~session_stats['is_abnormal']]['video_id'].tolist()[:2]
    abnormal_sessions = session_stats[session_stats['is_abnormal']]['video_id'].tolist()[:2]
    
    # Combine
    selected_sessions = normal_sessions + abnormal_sessions
    
    if len(selected_sessions) < 4:
        print("⚠️  Not enough sessions for comparison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Flow Maps: Normal vs Abnormal Recipe Execution',
                fontsize=20, fontweight='bold')
    
    for idx, video_id in enumerate(selected_sessions):
        ax = axes[idx // 2, idx % 2]
        
        # Get actions
        video_actions = recipe_narrations[recipe_narrations['video_id'] == video_id].copy()
        video_actions = video_actions.sort_values('start_timestamp')
        
        # Build mini graph
        G = nx.DiGraph()
        
        actions = []
        for _, row in video_actions.iterrows():
            if row['main_action_classes'] and len(row['main_action_classes']) > 0:
                verb_class, noun_class = row['main_action_classes'][0]
                action = get_action_name(verb_class, noun_class, verb_classes, noun_classes)
                actions.append(action)
        
        # Add edges
        for i in range(len(actions) - 1):
            if G.has_edge(actions[i], actions[i+1]):
                G[actions[i]][actions[i+1]]['weight'] += 1
            else:
                G.add_edge(actions[i], actions[i+1], weight=1)
        
        # Check if abnormal
        is_abnormal = video_id in abnormal_sessions
        
        # Layout
        pos = nx.spring_layout(G, k=2, seed=42)
        
        # Draw
        node_color = 'lightcoral' if is_abnormal else 'lightblue'
        edge_color = 'purple' if is_abnormal else 'gray'
        
        nx.draw(G, pos,
               node_color=node_color,
               edge_color=edge_color,
               node_size=500,
               width=2,
               with_labels=True,
               font_size=7,
               font_weight='bold',
               arrows=True,
               arrowsize=15,
               ax=ax)
        
        # Title
        label = 'B (Abnormal)' if is_abnormal else 'A (Normal)'
        stats = session_stats[session_stats['video_id'] == video_id].iloc[0]
        
        ax.set_title(f'{label}: {video_id}\n'
                    f'Loops: {stats["loops"]}, Duration: {stats["duration"]:.0f}s',
                    fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../outputs/figures/flow_map_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Flow map comparison saved")
    plt.close()


def main():
    # Load data
    data = load_hd_epic_data('..')

    recipe_info, recipe_narrations = load_selected_recipe_files('../outputs')
    
    # Categorize sessions
    session_stats = categorize_sessions(recipe_narrations, recipe_info['video_ids'])
    
    # Save stats
    session_stats.to_csv('../outputs/tables/session_stats.csv', index=False)
    print(f"\n✓ Session stats saved to ../outputs/tables/session_stats.csv")
    
    # Create flow maps
    create_flow_map_comparison(
        recipe_narrations,
        session_stats,
        data['verb_classes'],
        data['noun_classes']
    )
    
    print("\n" + "="*80)
    print("FLOW MAP VISUALIZATION COMPLETE")
    print("="*80)
    print("\nNext step: Run 5_bottleneck_analysis.py to find struggle points")

if __name__ == "__main__":
    main()