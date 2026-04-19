"""
Step 1.5: Hierarchical Action Abstraction
==========================================
THE MISSING PIECE: Why your visualizations were unreadable.

HD-EPIC has TWO annotation layers:
  1. Fine-grained narrations (PKL): ~84 actions per session
     e.g., open(cupboard), take(bowl), close(cupboard), take(oatmeal), ...
  2. High-level recipe steps (JSON): ~5-8 steps per recipe
     e.g., "Gather and prepare ingredients", "Add liquid", "Cook in microwave"

Your current pipeline uses ONLY layer 1, producing 157-node graphs.
This script MAPS layer 1 actions INTO layer 2 steps using temporal alignment.

The result: each fine-grained action is tagged with its parent high-level step,
allowing you to visualize at the recipe-step level (~8 nodes) instead of the
action level (~157 nodes).

This is the "semantic zoom" principle from Elmqvist & Fekete (2010).

Output:
  - outputs/tables/abstracted_traces_{recipe}.csv
  - Console: abstracted trace per session

Usage:
  python step1_5_hierarchical_abstraction.py coffee
  python step1_5_hierarchical_abstraction.py porridge
"""

import sys
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from utils import load_hd_epic_data, create_output_dirs, get_action_name


def load_recipe_data(recipe_query, recipes):
    """Find all recipe definitions matching the query."""
    query = recipe_query.strip().lower()
    matched = {}
    for rid, rdata in recipes.items():
        if query == rid.lower() or query in rdata.get('name', '').lower():
            matched[rid] = rdata
    return matched


def build_step_time_ranges(recipe_data):
    """
    From complete_recipes.json, extract the time ranges for each high-level step
    per video. This gives us the temporal boundaries for assigning fine-grained
    actions to high-level steps.

    Returns:
      dict: {video_id: [(step_id, step_label, start, end), ...]}
    """
    step_labels = recipe_data.get('steps', {})
    captures = recipe_data.get('captures', [])

    video_step_ranges = {}

    for capture in captures:
        for video_id in capture.get('videos', []):
            step_times = capture.get('step_times', {})
            prep_times = capture.get('prep_times', {})

            ranges = []
            for step_id, step_label in step_labels.items():
                # Collect all time segments for this step (step_times + prep_times)
                segments = []
                for seg in step_times.get(step_id, []):
                    if seg.get('video') == video_id:
                        segments.append((seg['start'], seg['end']))
                for seg in prep_times.get(step_id, []):
                    if seg.get('video') == video_id:
                        segments.append((seg['start'], seg['end']))

                if segments:
                    # Use the full range (min start to max end) for this step
                    overall_start = min(s[0] for s in segments)
                    overall_end = max(s[1] for s in segments)
                    ranges.append((step_id, step_label, overall_start, overall_end))

            # Sort by start time
            ranges.sort(key=lambda x: x[2])
            video_step_ranges[video_id] = ranges

    return video_step_ranges


def assign_actions_to_steps(narrations_df, video_step_ranges, verb_classes, noun_classes):
    """
    For each fine-grained narration, assign it to the closest high-level step
    based on temporal overlap.

    Strategy:
      1. If the action's midpoint falls within a step's time range → assign to that step
      2. If no overlap → assign to the nearest step by time proximity
      3. Actions before the first step → "PREP" phase
      4. Actions after the last step → "CLEANUP" phase
    """
    results = []

    for video_id in narrations_df['video_id'].unique():
        vid_df = narrations_df[narrations_df['video_id'] == video_id].sort_values('start_timestamp')
        step_ranges = video_step_ranges.get(video_id, [])

        if not step_ranges:
            # No high-level annotation for this video — assign all to "UNKNOWN"
            for _, row in vid_df.iterrows():
                mac = row.get('main_action_classes', None)
                if mac and len(mac) > 0:
                    vc, nc = mac[0]
                    action = get_action_name(vc, nc, verb_classes, noun_classes)
                    results.append({
                        'video_id': video_id,
                        'action': action,
                        'start': float(row['start_timestamp']),
                        'end': float(row['end_timestamp']),
                        'high_level_step_id': 'UNKNOWN',
                        'high_level_step_label': 'Unknown',
                        'assignment_method': 'no_annotation',
                    })
            continue

        for _, row in vid_df.iterrows():
            mac = row.get('main_action_classes', None)
            if not mac or len(mac) == 0:
                continue

            vc, nc = mac[0]
            action = get_action_name(vc, nc, verb_classes, noun_classes)
            action_start = float(row['start_timestamp'])
            action_end = float(row['end_timestamp'])
            action_mid = (action_start + action_end) / 2

            # Try temporal overlap first
            assigned = None
            for step_id, step_label, s_start, s_end in step_ranges:
                if s_start <= action_mid <= s_end:
                    assigned = (step_id, step_label, 'overlap')
                    break

            # If no overlap, find nearest step
            if assigned is None:
                min_dist = float('inf')
                nearest = None
                for step_id, step_label, s_start, s_end in step_ranges:
                    # Distance to the step's time range
                    if action_mid < s_start:
                        dist = s_start - action_mid
                    elif action_mid > s_end:
                        dist = action_mid - s_end
                    else:
                        dist = 0
                    if dist < min_dist:
                        min_dist = dist
                        nearest = (step_id, step_label, 'nearest')

                # If too far from any step, label as PREP or CLEANUP
                first_step_start = step_ranges[0][2]
                last_step_end = step_ranges[-1][3]

                if action_mid < first_step_start and min_dist > 30:
                    assigned = ('PREP', 'Preparation', 'before_recipe')
                elif action_mid > last_step_end and min_dist > 30:
                    assigned = ('CLEANUP', 'Clean up', 'after_recipe')
                else:
                    assigned = nearest

            results.append({
                'video_id': video_id,
                'action': action,
                'start': action_start,
                'end': action_end,
                'high_level_step_id': assigned[0],
                'high_level_step_label': assigned[1],
                'assignment_method': assigned[2],
            })

    return pd.DataFrame(results)


def build_abstracted_traces(assigned_df):
    """
    From the action-to-step assignments, build abstracted traces where each
    element is a high-level step (with counts of constituent actions).

    This is what goes into the canonical graph at the abstract level.
    """
    traces = []

    for video_id in assigned_df['video_id'].unique():
        vid_df = assigned_df[assigned_df['video_id'] == video_id].sort_values('start')

        # Build sequence of high-level steps (collapsing consecutive same-steps)
        step_sequence = []
        current_step = None
        current_actions = []

        for _, row in vid_df.iterrows():
            step_id = row['high_level_step_id']
            if step_id != current_step:
                if current_step is not None:
                    step_sequence.append({
                        'step_id': current_step,
                        'step_label': current_actions[0]['high_level_step_label'],
                        'actions': [a['action'] for a in current_actions],
                        'action_count': len(current_actions),
                        'start': current_actions[0]['start'],
                        'end': current_actions[-1]['end'],
                        'duration': current_actions[-1]['end'] - current_actions[0]['start'],
                    })
                current_step = step_id
                current_actions = [row.to_dict()]
            else:
                current_actions.append(row.to_dict())

        # Don't forget the last group
        if current_step is not None and current_actions:
            step_sequence.append({
                'step_id': current_step,
                'step_label': current_actions[0]['high_level_step_label'],
                'actions': [a['action'] for a in current_actions],
                'action_count': len(current_actions),
                'start': current_actions[0]['start'],
                'end': current_actions[-1]['end'],
                'duration': current_actions[-1]['end'] - current_actions[0]['start'],
            })

        traces.append({
            'video_id': video_id,
            'step_sequence': step_sequence,
            'step_ids': [s['step_id'] for s in step_sequence],
            'step_labels': [s['step_label'] for s in step_sequence],
            'total_actions': len(vid_df),
            'total_steps': len(step_sequence),
        })

    return traces


def print_trace_comparison(traces, step_labels):
    """Pretty-print the abstracted traces for comparison."""
    print(f"\n{'═'*80}")
    print("ABSTRACTED TRACE COMPARISON")
    print(f"{'═'*80}")

    # Print canonical step sequence as header
    print(f"\nCanonical steps: {list(step_labels.values())}")

    for trace in traces:
        vid = trace['video_id']
        steps = trace['step_sequence']
        print(f"\n{'─'*60}")
        print(f"  {vid}")
        print(f"  {trace['total_actions']} fine-grained actions → {trace['total_steps']} high-level steps")
        print(f"  Sequence: {' → '.join(trace['step_labels'])}")

        # Show each step with action count and duration
        for step in steps:
            dur = step['duration']
            print(f"    [{step['step_id']:15s}] {step['step_label']:40s} "
                  f"({step['action_count']:2d} actions, {dur:6.1f}s)")

        # Check for repeated steps (oscillation at abstract level)
        step_ids = trace['step_ids']
        for i in range(len(step_ids) - 2):
            if step_ids[i] == step_ids[i+2] and step_ids[i] != step_ids[i+1]:
                print(f"    ⚠ OSCILLATION: {step_ids[i]} → {step_ids[i+1]} → {step_ids[i]}")

        # Check for missing canonical steps
        canonical_ids = set(step_labels.keys())
        present_ids = set(step_ids)
        missing = canonical_ids - present_ids
        if missing:
            print(f"    ⚠ OMITTED STEPS: {', '.join(sorted(missing))}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python step1_5_hierarchical_abstraction.py <recipe_name_or_id>")
        sys.exit(1)

    recipe_query = sys.argv[1]

    data = load_hd_epic_data('..')
    create_output_dirs()

    # Find matching recipes
    matched = load_recipe_data(recipe_query, data['recipes'])
    if not matched:
        print(f"No recipe found matching '{recipe_query}'")
        sys.exit(1)

    print(f"\nFound {len(matched)} recipe definition(s) matching '{recipe_query}'")

    all_assigned = []
    all_traces = []

    for recipe_id, recipe_data in matched.items():
        recipe_name = recipe_data.get('name', 'Unknown')
        step_labels = recipe_data.get('steps', {})
        print(f"\n{'═'*80}")
        print(f"Processing: {recipe_id} — {recipe_name}")
        print(f"{'═'*80}")
        print(f"High-level steps:")
        for sid, slabel in step_labels.items():
            print(f"  {sid}: {slabel}")

        # Get time ranges for each step
        video_step_ranges = build_step_time_ranges(recipe_data)
        print(f"\nStep time ranges found for {len(video_step_ranges)} video(s)")

        # Get narrations for these videos
        all_videos = []
        for cap in recipe_data.get('captures', []):
            all_videos.extend(cap.get('videos', []))

        vid_narrations = data['narrations'][data['narrations']['video_id'].isin(all_videos)]
        print(f"Fine-grained narrations: {len(vid_narrations)} actions across {len(all_videos)} video(s)")

        if vid_narrations.empty:
            print("⚠️  No narrations found for these videos")
            continue

        # Assign actions to steps
        assigned_df = assign_actions_to_steps(
            vid_narrations, video_step_ranges,
            data['verb_classes'], data['noun_classes']
        )
        assigned_df['recipe_id'] = recipe_id
        all_assigned.append(assigned_df)

        # Build abstracted traces
        traces = build_abstracted_traces(assigned_df)
        all_traces.extend(traces)

        # Print comparison
        print_trace_comparison(traces, step_labels)

        # Assignment quality report
        print(f"\n--- Assignment Quality ---")
        method_counts = assigned_df['assignment_method'].value_counts()
        for method, count in method_counts.items():
            pct = count / len(assigned_df) * 100
            print(f"  {method:20s}: {count:4d} actions ({pct:.1f}%)")

    # Save outputs
    if all_assigned:
        combined_df = pd.concat(all_assigned, ignore_index=True)
        safe_name = recipe_query.lower().replace(' ', '_').replace(',', '')
        out_path = Path(f'../outputs/tables/abstracted_traces_{safe_name}.csv')
        combined_df.to_csv(out_path, index=False)
        print(f"\n✓ Abstracted assignments saved to {out_path}")

        # Also save traces as pickle for downstream use
        traces_path = Path(f'../outputs/graphs/abstracted_traces_{safe_name}.pkl')
        with open(traces_path, 'wb') as f:
            pickle.dump({
                'traces': all_traces,
                'recipe_query': recipe_query,
                'matched_recipes': {rid: rdata.get('name') for rid, rdata in matched.items()},
            }, f)
        print(f"✓ Abstracted traces saved to {traces_path}")

    print(f"\n{'═'*80}")
    print("HIERARCHICAL ABSTRACTION COMPLETE")
    print(f"{'═'*80}")
    print("\nYou can now run step2_canonical_graph.py using these abstracted traces")
    print("for a READABLE canonical graph (~8 nodes instead of ~157).")
    print("\nAlternatively, use the abstracted_traces CSV to build the interactive")
    print("React visualization (porridge_deviation_analysis.jsx).")


if __name__ == "__main__":
    main()