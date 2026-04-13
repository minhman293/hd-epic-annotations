"""
Utility functions for HD-EPIC motion graph analysis
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path

def load_hd_epic_data(data_dir='..'):
    """
    Load all necessary HD-EPIC files
    
    Returns:
        dict with keys: narrations, verb_classes, noun_classes, recipes, recipe_timestamps
    """
    data_dir = Path(data_dir)
    
    print("="*80)
    print("LOADING HD-EPIC DATASET")
    print("="*80)
    
    # Load narrations
    print("Loading narrations...")
    with open(data_dir / 'narrations-and-action-segments' / 'HD_EPIC_Narrations.pkl', 'rb') as f:
        narrations = pickle.load(f)
    narrations = pd.DataFrame(narrations)
    print(f"✓ Narrations loaded: {len(narrations)} actions")
    
    # Load verb classes
    print("Loading verb classes...")
    verb_classes = pd.read_csv(data_dir /  'narrations-and-action-segments' / 'HD_EPIC_verb_classes.csv')
    print(f"✓ Verb classes loaded: {len(verb_classes)} verbs")
    
    # Load noun classes
    print("Loading noun classes...")
    noun_classes = pd.read_csv(data_dir / 'narrations-and-action-segments' / 'HD_EPIC_noun_classes.csv')
    print(f"✓ Noun classes loaded: {len(noun_classes)} nouns")
    
    # Load recipes
    print("Loading recipes...")
    with open(data_dir / 'high-level' / 'complete_recipes.json', 'r') as f:
        recipes = json.load(f)
    print(f"✓ Recipes loaded: {len(recipes)} recipes")
    
    # Load all recipe timestamps
    print("Loading recipe timestamps...")
    recipe_timestamps = []
    activities_dir = data_dir / 'high-level' / 'activities'
    
    for csv_file in activities_dir.glob('P*_recipe_timestamps.csv'):
        df = pd.read_csv(csv_file)
        recipe_timestamps.append(df)
    
    recipe_timestamps = pd.concat(recipe_timestamps, ignore_index=True)
    print(f"✓ Recipe timestamps loaded: {len(recipe_timestamps)} entries")
    
    print("="*80)
    print("DATASET LOADED SUCCESSFULLY")
    print("="*80)
    
    return {
        'narrations': narrations,
        'verb_classes': verb_classes,
        'noun_classes': noun_classes,
        'recipes': recipes,
        'recipe_timestamps': recipe_timestamps
    }


def get_verb_name(verb_class_id, verb_classes_df):
    """Get verb name from verb class ID"""
    row = verb_classes_df[verb_classes_df['id'] == verb_class_id]
    if len(row) > 0:
        return row.iloc[0]['key']
    return f"verb_{verb_class_id}"


def get_noun_name(noun_class_id, noun_classes_df):
    """Get noun name from noun class ID"""
    row = noun_classes_df[noun_classes_df['id'] == noun_class_id]
    if len(row) > 0:
        return row.iloc[0]['key']
    return f"noun_{noun_class_id}"


def get_action_name(verb_class_id, noun_class_id, verb_classes_df, noun_classes_df):
    """Get full action name (verb + noun)"""
    verb = get_verb_name(verb_class_id, verb_classes_df)
    noun = get_noun_name(noun_class_id, noun_classes_df)
    return f"{verb}({noun})"


def count_loops(actions_series):
    """
    Count A → B → A oscillation patterns in action sequence
    
    Args:
        actions_series: pandas Series of action labels
        
    Returns:
        int: number of oscillation loops found
    """
    actions = actions_series.tolist()
    loop_count = 0
    
    for i in range(len(actions) - 2):
        if actions[i] == actions[i+2] and actions[i] != actions[i+1]:
            loop_count += 1
    
    return loop_count


def calculate_pause(narrations_df):
    """
    Calculate pause duration between consecutive actions
    
    Args:
        narrations_df: DataFrame with start_timestamp and end_timestamp columns
        
    Returns:
        Series with pause durations in seconds
    """
    # Sort by start time
    df = narrations_df.sort_values('start_timestamp').copy()
    
    # Calculate pause (gap between end of action i and start of action i+1)
    pauses = []
    
    for i in range(len(df) - 1):
        end_time = df.iloc[i]['end_timestamp']
        next_start = df.iloc[i+1]['start_timestamp']
        pause = next_start - end_time
        pauses.append(max(0, pause))  # Ensure non-negative
    
    pauses.append(0)  # Last action has no pause after it
    
    return pd.Series(pauses, index=df.index)


def create_output_dirs():
    """Create output directories if they don't exist"""
    Path('../outputs/graphs').mkdir(parents=True, exist_ok=True)
    Path('../outputs/tables').mkdir(parents=True, exist_ok=True)
    Path('../outputs/figures').mkdir(parents=True, exist_ok=True)