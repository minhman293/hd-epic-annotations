"""
Step 1: Load and explore HD-EPIC dataset
"""

import sys
from utils import load_hd_epic_data, create_output_dirs
import pandas as pd

def main():
    # Load data
    data = load_hd_epic_data('..')
    
    narrations = data['narrations']
    verb_classes = data['verb_classes']
    noun_classes = data['noun_classes']
    recipes = data['recipes']
    recipe_timestamps = data['recipe_timestamps']
    
    # Basic exploration
    print("\n" + "="*80)
    print("DATASET EXPLORATION")
    print("="*80)
    
    print(f"\nNarrations DataFrame shape: {narrations.shape}")
    print(f"Columns: {list(narrations.columns)}")
    
    print(f"\nUnique participants: {narrations['participant_id'].nunique()}")
    print(f"Unique videos: {narrations['video_id'].nunique()}")
    
    print(f"\nFirst few narrations:")
    print(narrations.head())
    
    # Explore recipes
    print("\n" + "="*80)
    print("AVAILABLE RECIPES")
    print("="*80)
    
    for recipe_id, recipe_info in list(recipes.items())[:10]:
        print(f"\n{recipe_id}:")
        print(f"  Name: {recipe_info.get('name', 'N/A')}")
        print(f"  Type: {recipe_info.get('type', 'N/A')}")
        print(f"  Steps: {len(recipe_info.get('steps', []))}")
    
    # Show recipe-video mapping
    print("\n" + "="*80)
    print("RECIPE-VIDEO MAPPING")
    print("="*80)
    print(recipe_timestamps.head(10))
    
    # Create output directories
    create_output_dirs()
    
    print("\n" + "="*80)
    print("DATA LOADING COMPLETE")
    print("="*80)
    print("\nNext step: Run 2_recipe_selector.py to choose a recipe for analysis")

if __name__ == "__main__":
    main()