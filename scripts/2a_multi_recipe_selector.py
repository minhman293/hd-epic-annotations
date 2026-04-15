"""
Step 2a: Select MULTIPLE recipes (3 simplest) and extract relevant videos
This creates a combined dataset for multi-recipe visualization
"""

from utils import load_hd_epic_data, create_output_dirs
import pandas as pd
import json
from pathlib import Path


def select_multiple_recipes(recipes, recipe_timestamps, narrations, num_recipes=3):
    """
    Select multiple simple recipes for combined visualization
    
    Args:
        recipes: Complete recipes dictionary
        recipe_timestamps: Recipe timestamp mappings
        narrations: All narrations
        num_recipes: Number of recipes to select (default 3)
    
    Returns:
        Dictionary with selected recipes info
    """
    
    print("\n" + "="*80)
    print(f"MULTI-RECIPE SELECTION (Top {num_recipes} Simplest)")
    print("="*80)
    
    # Normalize recipe IDs
    timestamps = recipe_timestamps.copy()
    timestamps['participant'] = timestamps['video_id'].astype(str).str.split('-').str[0]
    timestamps['recipe_id'] = timestamps['recipe_id'].fillna('').astype(str).str.strip()
    timestamps['full_recipe_id'] = timestamps.apply(
        lambda r: f"{r['participant']}_{r['recipe_id']}" if r['recipe_id'] else '',
        axis=1
    )
    
    # Build recipe info table
    recipe_info_list = []
    
    for recipe_id, recipe_data in recipes.items():
        recipe_rows = timestamps[timestamps['full_recipe_id'] == recipe_id]
        video_count = recipe_rows['video_id'].nunique()
        
        if video_count == 0:
            continue
        
        videos = recipe_rows['video_id'].tolist()
        action_count = len(narrations[narrations['video_id'].isin(videos)])
        
        recipe_info_list.append({
            'recipe_id': recipe_id,
            'name': recipe_data.get('name', 'N/A'),
            'videos': video_count,
            'actions': action_count,
            'steps': len(recipe_data.get('steps', [])),
            'recipe_data': recipe_data,
            'video_ids': videos
        })
    
    recipe_df = pd.DataFrame(recipe_info_list)
    
    # Sort by simplicity: fewest steps, then fewest actions
    recipe_df = recipe_df.sort_values(
        by=['steps', 'actions', 'videos'],
        ascending=[True, True, True]
    )
    
    print("\nAvailable recipes (sorted by simplicity):")
    print(recipe_df[['recipe_id', 'name', 'videos', 'actions', 'steps']].head(20).to_string(index=False))
    
    if len(recipe_df) < num_recipes:
        print(f"\n⚠️  Warning: Only {len(recipe_df)} recipes available, selecting all")
        num_recipes = len(recipe_df)
    
    # Select top N simplest recipes
    selected_recipes = recipe_df.head(num_recipes)
    
    print(f"\n" + "="*80)
    print(f"SELECTED {num_recipes} RECIPES")
    print("="*80)
    
    combined_info = {
        'num_recipes': num_recipes,
        'recipes': []
    }
    
    combined_narrations = []
    
    for idx, row in selected_recipes.iterrows():
        recipe_id = row['recipe_id']
        recipe_data = row['recipe_data']
        video_ids = row['video_ids']
        
        print(f"\n{idx + 1}. Recipe: {recipe_id}")
        print(f"   Name: {row['name']}")
        print(f"   Videos: {row['videos']}")
        print(f"   Actions: {row['actions']}")
        print(f"   Steps: {row['steps']}")
        
        # Get narrations for this recipe
        recipe_narrations = narrations[narrations['video_id'].isin(video_ids)].copy()
        
        # Tag narrations with recipe_id for later identification
        recipe_narrations['source_recipe'] = recipe_id
        
        combined_narrations.append(recipe_narrations)
        
        # Store recipe info
        combined_info['recipes'].append({
            'recipe_id': recipe_id,
            'name': row['name'],
            'videos': row['videos'],
            'actions': row['actions'],
            'steps': row['steps'],
            'recipe_data': recipe_data,
            'video_ids': video_ids
        })
    
    # Combine all narrations
    all_narrations = pd.concat(combined_narrations, ignore_index=True)
    
    print(f"\n" + "="*80)
    print("COMBINED DATASET STATISTICS")
    print("="*80)
    print(f"Total recipes: {num_recipes}")
    print(f"Total videos: {all_narrations['video_id'].nunique()}")
    print(f"Total actions: {len(all_narrations)}")
    
    # Save outputs
    outputs_dir = Path('../outputs')
    
    multi_recipe_json = outputs_dir / 'multi_recipe_selection.json'
    multi_recipe_narrations = outputs_dir / 'multi_recipe_narrations.pkl'
    
    with open(multi_recipe_json, 'w') as f:
        json.dump(combined_info, f, indent=2)
    
    all_narrations.to_pickle(multi_recipe_narrations)
    
    print(f"\n✓ Multi-recipe info saved to {multi_recipe_json}")
    print(f"✓ Combined narrations saved to {multi_recipe_narrations}")
    
    print("\n" + "="*80)
    print("MULTI-RECIPE SELECTION COMPLETE")
    print("="*80)
    print("\nNext step: Run 3a_multi_recipe_graph.py to build combined motion graph")
    
    return combined_info, all_narrations


def main():
    data = load_hd_epic_data('..')
    create_output_dirs()
    
    # Select 3 simplest recipes
    select_multiple_recipes(
        data['recipes'],
        data['recipe_timestamps'],
        data['narrations'],
        num_recipes=3
    )


if __name__ == "__main__":
    main()