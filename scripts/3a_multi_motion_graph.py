"""
Step 3a: Build multi-recipe motion graph
Shows multiple recipe paths through shared action space
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
import pickle
from pathlib import Path
from collections import defaultdict

from utils import (load_hd_epic_data, get_verb_name, get_noun_name, 
                   get_action_name, calculate_pause)
from utils_multi import (get_pause_color, get_recipe_linestyle,
                         create_multi_recipe_legend, truncate_label,
                         calculate_node_positions_hierarchical,
                         draw_multi_recipe_edge)


def build_multi_recipe_graph(narrations_df, verb_classes_df, noun_classes_df):
    """
    Build motion graph from multiple recipes
    Each edge tracks which recipe(s) use it
    
    Returns:
        G: NetworkX MultiDiGraph
        recipe_edge_map: Dict mapping (u, v) -> list of recipe_ids
    """
    
    G = nx.MultiDiGraph()
    recipe_edge_map = defaultdict(list)
    
    print("\n" + "="*80)
    print("BUILDING MULTI-RECIPE MOTION GRAPH")
    print("="*80)
    
    # Process each video, tracking source recipe
    for video_id in narrations_df['video_id'].unique():
        video_actions = narrations_df[narrations_df['video_id'] == video_id].sort_values('start_timestamp')
        
        if len(video_actions) < 2:
            continue
        
        # Get recipe ID for this video
        source_recipe = video_actions.iloc[0].get('source_recipe', 'unknown')
        
        # Calculate pauses
        video_actions = video_actions.copy()
        video_actions['pause_after'] = calculate_pause(video_actions)
        
        # Extract actions
        actions = []
        pauses = []
        
        for idx, row in video_actions.iterrows():
            if row['main_action_classes'] and len(row['main_action_classes']) > 0:
                verb_class, noun_class = row['main_action_classes'][0]
                action_name = get_action_name(verb_class, noun_class, 
                                              verb_classes_df, noun_classes_df)
                actions.append(action_name)
                pauses.append(row['pause_after'])
        
        # Build edges
        for i in range(len(actions) - 1):
            action_a = actions[i]
            action_b = actions[i + 1]
            pause = pauses[i]
            
            # Track which recipe uses this edge
            edge_key = (action_a, action_b)
            if source_recipe not in recipe_edge_map[edge_key]:
                recipe_edge_map[edge_key].append(source_recipe)
            
            # Add/update edge
            if G.has_edge(action_a, action_b):
                # Find edge for this recipe
                edge_found = False
                for key, data in G[action_a][action_b].items():
                    if data.get('recipe_id') == source_recipe:
                        # Update existing edge for this recipe
                        data['weight'] += 1
                        data['pauses'].append(pause)
                        edge_found = True
                        break
                
                if not edge_found:
                    # Create new edge for this recipe
                    G.add_edge(action_a, action_b, 
                              recipe_id=source_recipe,
                              weight=1, 
                              pauses=[pause])
            else:
                # First edge between these nodes
                G.add_edge(action_a, action_b,
                          recipe_id=source_recipe,
                          weight=1,
                          pauses=[pause])
    
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
    print(f"  Recipes: {narrations_df['source_recipe'].nunique()}")
    
    # Show shared vs unique edges
    shared_edges = sum(1 for edge_recipes in recipe_edge_map.values() if len(edge_recipes) > 1)
    unique_edges = sum(1 for edge_recipes in recipe_edge_map.values() if len(edge_recipes) == 1)
    
    print(f"\n  Shared transitions (used by multiple recipes): {shared_edges}")
    print(f"  Unique transitions (recipe-specific): {unique_edges}")
    
    return G, recipe_edge_map


def visualize_multi_recipe_graph(G, recipe_edge_map, recipes_info, 
                                 output_path='../outputs/figures/multi_recipe_graph.png'):
    """
    Visualize multi-recipe motion graph
    - Different line styles for different recipes
    - Colors for pause duration
    - Node size for frequency
    """
    
    print("\n" + "="*80)
    print("VISUALIZING MULTI-RECIPE GRAPH")
    print("="*80)
    
    fig, ax = plt.subplots(figsize=(26, 22))
    
    # Calculate positions using hierarchical layout
    print("Computing layout...")
    pos = calculate_node_positions_hierarchical(G, recipes_info)
    
    # Calculate node sizes based on total degree
    print("Calculating node sizes...")
    node_degrees = dict(G.degree(weight='weight'))
    max_degree = max(node_degrees.values()) if node_degrees else 1
    node_sizes = [node_degrees.get(node, 1) / max_degree * 2500 for node in G.nodes()]
    
    # Draw nodes
    print("Drawing nodes...")
    nx.draw_networkx_nodes(G, pos,
                          node_size=node_sizes,
                          node_color='#93C5FD',
                          alpha=0.9,
                          edgecolors='#1E3A8A',
                          linewidths=2.5,
                          ax=ax)
    
    # Draw edges grouped by recipe
    print("Drawing edges...")
    
    # Create recipe ID to index mapping
    recipe_ids = [r['recipe_id'] for r in recipes_info]
    recipe_to_index = {rid: i for i, rid in enumerate(recipe_ids)}
    
    edge_count = 0
    for u, v, key, data in G.edges(data=True, keys=True):
        recipe_id = data.get('recipe_id', 'unknown')
        recipe_index = recipe_to_index.get(recipe_id, 0)
        
        draw_multi_recipe_edge(ax, pos, u, v, data, recipe_index)
        edge_count += 1
    
    print(f"  Drew {edge_count} edges")
    
    # Draw labels
    print("Drawing labels...")
    labels = {node: truncate_label(node, max_length=18) for node in G.nodes()}
    
    nx.draw_networkx_labels(G, pos, labels,
                           font_size=8,
                           font_weight='bold',
                           font_color='#0F172A',
                           ax=ax)
    
    # Title
    recipe_names = [r['name'] for r in recipes_info]
    recipes_str = ' + '.join(recipe_names)
    
    ax.set_title(
        f'Multi-Recipe Motion Graph: {recipes_str}\n'
        f'Different line styles = different recipe paths | Colors = pause duration',
        fontsize=18,
        fontweight='bold',
        pad=25
    )
    
    ax.axis('off')
    
    # Legend
    print("Creating legend...")
    legend_handles = create_multi_recipe_legend(recipe_names)
    ax.legend(handles=legend_handles, loc='upper right', fontsize=11, 
             frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Multi-recipe graph saved to {output_path}")
    
    plt.close()


def analyze_recipe_overlap(recipe_edge_map, recipes_info):
    """
    Analyze which actions/transitions are shared vs unique
    """
    
    print("\n" + "="*80)
    print("RECIPE OVERLAP ANALYSIS")
    print("="*80)
    
    # Create mapping
    recipe_ids = [r['recipe_id'] for r in recipes_info]
    recipe_names = {r['recipe_id']: r['name'] for r in recipes_info}
    
    # Categorize edges
    shared_by_all = []
    shared_by_some = []
    unique_per_recipe = {rid: [] for rid in recipe_ids}
    
    for (u, v), recipe_list in recipe_edge_map.items():
        if len(recipe_list) == len(recipe_ids):
            shared_by_all.append((u, v))
        elif len(recipe_list) > 1:
            shared_by_some.append((u, v, recipe_list))
        else:
            recipe_id = recipe_list[0]
            unique_per_recipe[recipe_id].append((u, v))
    
    print(f"\nTransitions shared by ALL recipes: {len(shared_by_all)}")
    if shared_by_all:
        print("  Examples:")
        for u, v in shared_by_all[:5]:
            print(f"    {u} → {v}")
    
    print(f"\nTransitions shared by SOME recipes: {len(shared_by_some)}")
    if shared_by_some:
        print("  Examples:")
        for u, v, rids in shared_by_some[:5]:
            recipe_str = ', '.join([recipe_names[r] for r in rids])
            print(f"    {u} → {v}")
            print(f"      Used by: {recipe_str}")
    
    print(f"\nUnique transitions per recipe:")
    for recipe_id, edges in unique_per_recipe.items():
        print(f"  {recipe_names[recipe_id]}: {len(edges)} unique transitions")
        if edges:
            print(f"    Examples:")
            for u, v in edges[:3]:
                print(f"      {u} → {v}")
    
    # Save analysis
    analysis = {
        'shared_by_all': [(u, v) for u, v in shared_by_all],
        'shared_by_some': [
            {'transition': (u, v), 'recipes': rids}
            for u, v, rids in shared_by_some
        ],
        'unique_per_recipe': {
            recipe_names[rid]: [(u, v) for u, v in edges]
            for rid, edges in unique_per_recipe.items()
        }
    }
    
    output_path = Path('../outputs/tables/recipe_overlap_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n✓ Overlap analysis saved to {output_path}")


def save_graph_data(G, recipe_edge_map, output_path='../outputs/graphs/multi_recipe_graph.pkl'):
    """Save graph and recipe mapping"""
    data = {
        'graph': G,
        'recipe_edge_map': dict(recipe_edge_map)
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✓ Multi-recipe graph data saved to {output_path}")


def main():
    # Load data
    data = load_hd_epic_data('..')
    
    # Load multi-recipe selection
    multi_recipe_json = Path('../outputs/multi_recipe_selection.json')
    multi_recipe_narrations = Path('../outputs/multi_recipe_narrations.pkl')
    
    if not multi_recipe_json.exists():
        raise FileNotFoundError(
            "Multi-recipe selection not found. Run 2a_multi_recipe_selector.py first."
        )
    
    with open(multi_recipe_json, 'r') as f:
        multi_recipe_info = json.load(f)
    
    narrations = pd.read_pickle(multi_recipe_narrations)
    
    print(f"\nAnalyzing {multi_recipe_info['num_recipes']} recipes:")
    for r in multi_recipe_info['recipes']:
        print(f"  - {r['recipe_id']}: {r['name']} ({r['actions']} actions)")
    
    # Build graph
    G, recipe_edge_map = build_multi_recipe_graph(
        narrations,
        data['verb_classes'],
        data['noun_classes']
    )
    
    # Visualize
    visualize_multi_recipe_graph(
        G,
        recipe_edge_map,
        multi_recipe_info['recipes']
    )
    
    # Analyze overlap
    analyze_recipe_overlap(recipe_edge_map, multi_recipe_info['recipes'])
    
    # Save
    save_graph_data(G, recipe_edge_map)
    
    print("\n" + "="*80)
    print("MULTI-RECIPE MOTION GRAPH COMPLETE")
    print("="*80)
    print("\nKey insights:")
    print("  - Different line styles show different recipe paths")
    print("  - Edge colors show pause duration (struggle points)")
    print("  - Node size shows action frequency across all recipes")
    print("  - Shared transitions indicate common cooking patterns")
    print("\nNext: Analyze bottlenecks and intervention points")


if __name__ == "__main__":
    main()