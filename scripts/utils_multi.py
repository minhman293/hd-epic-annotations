"""
Utility functions for multi-recipe motion graph analysis
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Define distinct line styles for different recipes
RECIPE_LINE_STYLES = [
    {'linestyle': '-', 'name': 'solid'},      # Recipe 1
    {'linestyle': '--', 'name': 'dashed'},    # Recipe 2  
    {'linestyle': ':', 'name': 'dotted'},     # Recipe 3
    {'linestyle': '-.', 'name': 'dashdot'},   # Recipe 4 (if needed)
]


# Define pause-based colors (same as before)
def get_pause_color(avg_pause):
    """Get edge color based on pause duration"""
    if avg_pause > 30:
        return '#DC2626'  # Red
    elif avg_pause > 10:
        return '#F59E0B'  # Orange
    else:
        return '#16A34A'  # Green


def get_recipe_linestyle(recipe_index):
    """Get line style for a recipe based on its index"""
    if recipe_index < len(RECIPE_LINE_STYLES):
        return RECIPE_LINE_STYLES[recipe_index]['linestyle']
    else:
        # Cycle through if more recipes than styles
        return RECIPE_LINE_STYLES[recipe_index % len(RECIPE_LINE_STYLES)]['linestyle']


def create_multi_recipe_legend(recipe_names):
    """
    Create legend showing both recipe paths and pause categories
    
    Args:
        recipe_names: List of recipe names
        
    Returns:
        List of legend handles
    """
    legend_handles = []
    
    # Recipe path styles
    legend_handles.append(mpatches.Patch(color='none', label='Recipe Paths:'))
    
    for i, name in enumerate(recipe_names):
        linestyle = get_recipe_linestyle(i)
        
        # Create a line patch with the style
        from matplotlib.lines import Line2D
        line = Line2D([0], [0], color='gray', linewidth=3, 
                     linestyle=linestyle, label=f'  {name}')
        legend_handles.append(line)
    
    # Spacer
    legend_handles.append(mpatches.Patch(color='none', label=''))
    
    # Pause categories
    legend_handles.append(mpatches.Patch(color='none', label='Pause Duration:'))
    
    from matplotlib.lines import Line2D
    legend_handles.append(Line2D([0], [0], color='#16A34A', linewidth=3, 
                                label='  Fast (<10s)'))
    legend_handles.append(Line2D([0], [0], color='#F59E0B', linewidth=3,
                                label='  Medium (10-30s)'))
    legend_handles.append(Line2D([0], [0], color='#DC2626', linewidth=3,
                                label='  Slow (>30s)'))
    
    return legend_handles


def truncate_label(label, max_length=20):
    """Truncate long action labels"""
    if len(label) <= max_length:
        return label
    return label[:max_length-3] + '...'


def calculate_node_positions_hierarchical(G, recipes_info):
    """
    Calculate node positions using hierarchical layout
    Groups nodes by typical recipe phase
    
    Args:
        G: NetworkX graph
        recipes_info: List of recipe information dictionaries
        
    Returns:
        Dictionary of node positions
    """
    import networkx as nx
    
    # Categorize nodes by action type
    prep_actions = []
    cook_actions = []
    serve_actions = []
    cleanup_actions = []
    other_actions = []
    
    for node in G.nodes():
        node_lower = node.lower()
        
        # Simple heuristic categorization
        if any(word in node_lower for word in ['open', 'take', 'get', 'pick', 'remove', 'peel']):
            prep_actions.append(node)
        elif any(word in node_lower for word in ['cook', 'fry', 'boil', 'stir', 'mix', 'pour', 'add', 'put']):
            cook_actions.append(node)
        elif any(word in node_lower for word in ['serve', 'plate', 'transfer']):
            serve_actions.append(node)
        elif any(word in node_lower for word in ['wash', 'clean', 'wipe', 'dry', 'close']):
            cleanup_actions.append(node)
        else:
            other_actions.append(node)
    
    # Create shells (concentric circles)
    shells = []
    if prep_actions:
        shells.append(prep_actions)
    if cook_actions:
        shells.append(cook_actions)
    if serve_actions:
        shells.append(serve_actions)
    if cleanup_actions:
        shells.append(cleanup_actions)
    if other_actions:
        shells.append(other_actions)
    
    # Use shell layout
    if shells:
        pos = nx.shell_layout(G, shells)
    else:
        # Fallback to spring layout
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    return pos


def draw_multi_recipe_edge(ax, pos, u, v, edge_data, recipe_index):
    """
    Draw a single edge with recipe-specific style and pause-based color
    
    Args:
        ax: Matplotlib axes
        pos: Node positions dictionary
        u, v: Source and target nodes
        edge_data: Edge data dictionary
        recipe_index: Index of recipe (for line style)
    """
    from matplotlib.patches import FancyArrowPatch
    
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    
    # Get style
    linestyle = get_recipe_linestyle(recipe_index)
    color = get_pause_color(edge_data['avg_pause'])
    
    # Width based on frequency
    max_weight = 10  # Normalize
    width = 1.5 + (edge_data['weight'] / max_weight) * 4
    
    # Draw arrow
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->',
        linestyle=linestyle,
        color=color,
        linewidth=width,
        alpha=0.7,
        mutation_scale=20,
        connectionstyle='arc3,rad=0.1'
    )
    
    ax.add_patch(arrow)