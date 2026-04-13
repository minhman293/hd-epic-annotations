"""
Step 3: Build motion graph from recipe actions
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
import pickle
from utils import (load_hd_epic_data, get_verb_name, get_noun_name, 
                   get_action_name, calculate_pause)

def build_motion_graph(narrations_df, verb_classes_df, noun_classes_df):
    """
    Build motion graph where:
    - Nodes = Actions (verb-noun pairs)
    - Edges = Consecutive actions  
    - Edge attributes = frequency and pause statistics
    """
    
    G = nx.MultiDiGraph()
    
    print("\n" + "="*80)
    print("BUILDING MOTION GRAPH")
    print("="*80)
    
    # Process each video separately
    for video_id in narrations_df['video_id'].unique():
        video_actions = narrations_df[narrations_df['video_id'] == video_id].sort_values('start_timestamp')
        
        if len(video_actions) < 2:
            continue
        
        # Calculate pauses
        video_actions['pause_after'] = calculate_pause(video_actions)
        
        # Extract main actions (verb_class, noun_class from main_action_classes)
        actions = []
        times_start = []
        times_end = []
        pauses = []
        
        for idx, row in video_actions.iterrows():
            # Get main action (first pair)
            if row['main_action_classes'] and len(row['main_action_classes']) > 0:
                verb_class, noun_class = row['main_action_classes'][0]
                
                action_name = get_action_name(verb_class, noun_class, 
                                              verb_classes_df, noun_classes_df)
                
                actions.append(action_name)
                times_start.append(row['start_timestamp'])
                times_end.append(row['end_timestamp'])
                pauses.append(row['pause_after'])
        
        # Build edges
        for i in range(len(actions) - 1):
            action_a = actions[i]
            action_b = actions[i + 1]
            pause = pauses[i]
            
            # Add edge
            if G.has_edge(action_a, action_b):
                # Find existing edge with same endpoints
                edge_key = list(G[action_a][action_b].keys())[0]
                G[action_a][action_b][edge_key]['weight'] += 1
                G[action_a][action_b][edge_key]['pauses'].append(pause)
            else:
                G.add_edge(action_a, action_b, weight=1, pauses=[pause])
    
    # Calculate statistics for each edge
    for u, v, key, data in G.edges(data=True, keys=True):
        pauses = data['pauses']
        data['avg_pause'] = np.mean(pauses)
        data['max_pause'] = np.max(pauses)
        data['min_pause'] = np.min(pauses)
        data['std_pause'] = np.std(pauses)
    
    print(f"\nGraph statistics:")
    print(f"  Nodes (unique actions): {G.number_of_nodes()}")
    print(f"  Edges (transitions): {G.number_of_edges()}")
    print(f"  Videos analyzed: {narrations_df['video_id'].nunique()}")
    
    return G


def visualize_motion_graph(G, output_path='../outputs/figures/motion_graph.png'):
    """
    Visualize motion graph like Figure 10 example
    """
    
    print("\n" + "="*80)
    print("VISUALIZING MOTION GRAPH")
    print("="*80)
    
    fig, ax = plt.subplots(figsize=(24, 20))
    
    # Layout - spring for better node distribution
    print("Computing layout...")
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    # Node sizes based on degree (frequency)
    print("Calculating node sizes...")
    node_degrees = dict(G.degree(weight='weight'))
    max_degree = max(node_degrees.values()) if node_degrees else 1
    node_sizes = [node_degrees.get(node, 1) / max_degree * 2000 for node in G.nodes()]
    
    # Draw nodes
    print("Drawing nodes...")
    nx.draw_networkx_nodes(G, pos,
                          node_size=node_sizes,
                          node_color='lightblue',
                          alpha=0.8,
                          edgecolors='black',
                          linewidths=2,
                          ax=ax)
    
    # Prepare edges
    print("Processing edges...")
    edge_data = []
    for u, v, key, data in G.edges(data=True, keys=True):
        edge_data.append({
            'u': u,
            'v': v,
            'weight': data['weight'],
            'avg_pause': data['avg_pause']
        })
    
    # Normalize edge widths
    max_weight = max([e['weight'] for e in edge_data]) if edge_data else 1
    
    # Draw edges by pause category
    print("Drawing edges...")
    for edge in edge_data:
        width = (edge['weight'] / max_weight) * 8
        
        # Color by pause duration
        if edge['avg_pause'] > 30:
            color = 'red'
        elif edge['avg_pause'] > 10:
            color = 'orange'
        else:
            color = 'green'
        
        nx.draw_networkx_edges(G, pos,
                              edgelist=[(edge['u'], edge['v'])],
                              width=width,
                              edge_color=color,
                              alpha=0.6,
                              arrows=True,
                              arrowsize=20,
                              arrowstyle='->',
                              connectionstyle='arc3,rad=0.1',
                              ax=ax)
    
    # Draw labels
    print("Drawing labels...")
    # Truncate long labels
    labels = {node: node[:20] + '...' if len(node) > 20 else node 
             for node in G.nodes()}
    
    nx.draw_networkx_labels(G, pos, labels,
                           font_size=8,
                           font_weight='bold',
                           ax=ax)
    
    ax.set_title('Recipe Motion Graph: Action Flow with Pause Indicators',
                fontsize=20, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', linewidth=4, 
               label='Fast transition (<10s pause)'),
        Line2D([0], [0], color='orange', linewidth=4,
               label='Medium pause (10-30s)'),
        Line2D([0], [0], color='red', linewidth=4,
               label='Long pause (>30s) - Struggle point')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Motion graph saved to {output_path}")
    
    plt.close()


def save_graph_data(G, output_path='../outputs/graphs/motion_graph.pkl'):
    """Save graph object for later use"""
    with open(output_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"✓ Graph data saved to {output_path}")


def main():
    # Load data
    data = load_hd_epic_data('..')
    
    # Load recipe selection
    with open('../outputs/selected_recipe.json', 'r') as f:
        recipe_info = json.load(f)
    
    recipe_narrations = pd.read_pickle('../outputs/recipe_narrations.pkl')
    
    print(f"\nAnalyzing recipe: {recipe_info['recipe_id']}")
    print(f"Videos: {len(recipe_info['video_ids'])}")
    print(f"Actions: {recipe_info['narrations_count']}")
    
    # Build graph
    G = build_motion_graph(
        recipe_narrations,
        data['verb_classes'],
        data['noun_classes']
    )
    
    # Visualize
    visualize_motion_graph(G)
    
    # Save
    save_graph_data(G)
    
    print("\n" + "="*80)
    print("MOTION GRAPH COMPLETE")
    print("="*80)
    print("\nNext step: Run 4_visualize_flow_maps.py for flow map analysis")

if __name__ == "__main__":
    main()