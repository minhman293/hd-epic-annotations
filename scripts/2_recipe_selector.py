"""
Step 2: Select a recipe and extract relevant videos
"""

from utils import load_hd_epic_data, create_output_dirs
import pandas as pd
import json

def select_recipe(recipes, recipe_timestamps, narrations):
    """
    Interactive recipe selection
    """
    
    print("\n" + "="*80)
    print("RECIPE SELECTION")
    print("="*80)
    
    # Normalize activity recipe IDs (e.g. R01) to full IDs (e.g. P01_R01)
    timestamps = recipe_timestamps.copy()
    timestamps['participant'] = timestamps['video_id'].astype(str).str.split('-').str[0]
    timestamps['recipe_id'] = timestamps['recipe_id'].fillna('').astype(str).str.strip()
    timestamps['full_recipe_id'] = timestamps.apply(
        lambda r: f"{r['participant']}_{r['recipe_id']}" if r['recipe_id'] else '',
        axis=1
    )

    # Show available recipes with video counts
    recipe_info = []
    
    for recipe_id, recipe_data in recipes.items():
        # Count videos for this recipe
        recipe_rows = timestamps[timestamps['full_recipe_id'] == recipe_id]
        video_count = recipe_rows['video_id'].nunique()
        
        # Count total actions
        videos = recipe_rows['video_id'].tolist()
        action_count = len(narrations[narrations['video_id'].isin(videos)])
        
        recipe_info.append({
            'recipe_id': recipe_id,
            'name': recipe_data.get('name', 'N/A'),
            'videos': video_count,
            'actions': action_count,
            'steps': len(recipe_data.get('steps', []))
        })
    
    recipe_df = pd.DataFrame(recipe_info)
    recipe_df = recipe_df[recipe_df['videos'] > 0].sort_values(
        by=['videos', 'actions', 'steps', 'recipe_id'],
        ascending=[True, True, True, True]
    )
    
    print("\nTop simplest recipes (fewest videos/actions/steps):")
    print(recipe_df.head(20).to_string(index=False))

    if recipe_df.empty:
        raise ValueError(
            "No recipes matched between complete_recipes.json and activity timestamps. "
            "Check recipe_id formatting in activities CSV files."
        )
    
    # Auto-select simplest recipe based on fewest videos, actions, then steps
    selected_row = recipe_df.iloc[0]
    selected_recipe_id = selected_row['recipe_id']
    
    print(f"\n" + "="*80)
    print(f"SELECTED RECIPE: {selected_recipe_id}")
    print(f"="*80)
    
    recipe_data = recipes[selected_recipe_id]
    
    print(f"\nName: {recipe_data.get('name')}")
    print(f"Type: {recipe_data.get('type')}")
    print(f"Videos: {selected_row['videos']}")
    print(f"Total actions: {selected_row['actions']}")
    
    print(f"\nIdeal recipe steps:")
    steps = recipe_data.get('steps', {})
    for i, (_, step_text) in enumerate(steps.items(), 1):
        print(f"  {i}. {step_text}")
    
    # Get videos for this recipe
    recipe_videos = timestamps[
        timestamps['full_recipe_id'] == selected_recipe_id
    ]['video_id'].tolist()
    
    print(f"\nVideos containing this recipe:")
    for vid in recipe_videos:
        action_count = len(narrations[narrations['video_id'] == vid])
        print(f"  - {vid}: {action_count} actions")
    
    # Extract narrations for these videos
    recipe_narrations = narrations[narrations['video_id'].isin(recipe_videos)].copy()
    
    # Save for next steps
    output_data = {
        'recipe_id': selected_recipe_id,
        'recipe_data': recipe_data,
        'video_ids': recipe_videos,
        'narrations_count': len(recipe_narrations)
    }
    
    with open('../outputs/selected_recipe.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    recipe_narrations.to_pickle('../outputs/recipe_narrations.pkl')
    
    print(f"\n✓ Recipe data saved to ../outputs/selected_recipe.json")
    print(f"✓ Narrations saved to ../outputs/recipe_narrations.pkl")
    
    print("\n" + "="*80)
    print("RECIPE SELECTION COMPLETE")
    print("="*80)
    print("\nNext step: Run 3_motion_graph.py to build the motion graph")
    
    return selected_recipe_id, recipe_data, recipe_videos, recipe_narrations


def main():
    data = load_hd_epic_data('..')
    create_output_dirs()
    
    select_recipe(
        data['recipes'],
        data['recipe_timestamps'],
        data['narrations']
    )

if __name__ == "__main__":
    main()