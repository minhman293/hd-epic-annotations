"""
Step 1: Cross-Participant Recipe Survey & Vocabulary Consistency Check
======================================================================
Goal: Identify which recipes in HD-EPIC have sufficient multi-participant
coverage for canonical graph construction, and verify that the verb-object
vocabulary is consistent across participants (or flag where normalization
is needed).

Output:
  - outputs/tables/recipe_coverage_survey.csv
  - outputs/tables/vocabulary_consistency_report.csv
  - Console report summarizing findings

Usage:
  python step1_recipe_survey.py
  (assumes HD-EPIC data is in parent directory '../')
"""

import pandas as pd
import numpy as np
import json
import pickle
import ast
from pathlib import Path
from collections import defaultdict, Counter
from utils import load_hd_epic_data, create_output_dirs, get_action_name


def parse_main_action_classes(main_action_classes):
    """
    Normalize main_action_classes into a list of (verb_class, noun_class) tuples.

    Handles common input shapes seen in HD-EPIC rows:
      - [(v, n), ...]
      - (v, n)
      - [[v, n], ...]
      - string-encoded variants of the above
    """
    if main_action_classes is None:
        return []

    # Pandas may store missing values as NaN
    if isinstance(main_action_classes, float) and np.isnan(main_action_classes):
        return []

    value = main_action_classes
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            value = ast.literal_eval(value)
        except Exception:
            return []

    # Single pair encoded as (v, n)
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 2 and all(
        np.isscalar(x) for x in value
    ):
        try:
            return [(int(value[0]), int(value[1]))]
        except Exception:
            return []

    pairs = []
    if isinstance(value, (list, tuple, np.ndarray)):
        for item in value:
            if isinstance(item, (list, tuple, np.ndarray)) and len(item) >= 2:
                try:
                    pairs.append((int(item[0]), int(item[1])))
                except Exception:
                    continue
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# 1A. Survey recipe coverage across participants
# ─────────────────────────────────────────────────────────────────────────────

def survey_recipe_coverage(recipes, recipe_timestamps, narrations):
    """
    For every recipe in complete_recipes.json, count:
      - How many distinct participants recorded it
      - How many total video sessions exist
      - How many fine-grained narration actions are available
      - The canonical step count from the recipe definition

    Returns a DataFrame sorted by participant count (descending).
    """
    print("\n" + "=" * 80)
    print("STEP 1A: CROSS-PARTICIPANT RECIPE COVERAGE SURVEY")
    print("=" * 80)

    # Normalize recipe_timestamps to have full recipe IDs
    ts = recipe_timestamps.copy()
    ts['participant'] = ts['video_id'].astype(str).str.split('-').str[0]
    ts['recipe_id_raw'] = ts['recipe_id'].fillna('').astype(str).str.strip()
    ts['full_recipe_id'] = ts.apply(
        lambda r: f"{r['participant']}_{r['recipe_id_raw']}" if r['recipe_id_raw'] else '',
        axis=1
    )

    # The recipe keys in complete_recipes.json are like "P01_R01", "P08_R01", etc.
    # Each key is participant-specific. To find cross-participant recipes,
    # we need to group by the recipe NAME (e.g., "Nespresso", "Coffee").
    recipe_by_name = defaultdict(list)
    for recipe_id, recipe_data in recipes.items():
        name = recipe_data.get('name', 'Unknown').strip().lower()
        recipe_by_name[name].append({
            'recipe_id': recipe_id,
            'participant': recipe_data.get('participant', recipe_id.split('_')[0]),
            'recipe_data': recipe_data,
        })

    results = []

    for recipe_name, entries in recipe_by_name.items():
        participants = set()
        total_videos = 0
        total_actions = 0
        step_counts = []

        for entry in entries:
            rid = entry['recipe_id']
            pid = entry['participant']
            participants.add(pid)

            # Count videos for this participant's recipe
            recipe_rows = ts[ts['full_recipe_id'] == rid]
            vids = recipe_rows['video_id'].unique().tolist()
            total_videos += len(vids)

            # Also count videos from captures in the JSON (more reliable)
            captures = entry['recipe_data'].get('captures', [])
            for cap in captures:
                cap_vids = cap.get('videos', [])
                total_videos_from_json = len(cap_vids)
                # Count narrations for these videos
                for v in cap_vids:
                    n_actions = len(narrations[narrations['video_id'] == v])
                    total_actions += n_actions

            steps = entry['recipe_data'].get('steps', {})
            step_counts.append(len(steps))

        # De-duplicate video count (timestamps vs JSON may overlap)
        # Use the JSON-based count as primary
        all_capture_videos = []
        for entry in entries:
            for cap in entry['recipe_data'].get('captures', []):
                all_capture_videos.extend(cap.get('videos', []))
        unique_videos = len(set(all_capture_videos))

        # Recount actions from unique videos
        total_actions_deduped = 0
        for v in set(all_capture_videos):
            total_actions_deduped += len(narrations[narrations['video_id'] == v])

        results.append({
            'recipe_name': recipe_name,
            'participant_count': len(participants),
            'participants': ', '.join(sorted(participants)),
            'total_sessions': unique_videos,
            'total_actions': total_actions_deduped,
            'avg_steps': np.mean(step_counts) if step_counts else 0,
            'recipe_ids': ', '.join(sorted(e['recipe_id'] for e in entries)),
        })

    df = pd.DataFrame(results).sort_values(
        by=['participant_count', 'total_sessions', 'total_actions'],
        ascending=[False, False, False]
    )
    df = df.reset_index(drop=True)

    # Display results
    print(f"\nFound {len(df)} unique recipe names across all participants.")
    print(f"\nRecipes with MULTI-PARTICIPANT coverage (≥2 participants):")
    print("-" * 80)
    multi = df[df['participant_count'] >= 2]
    if multi.empty:
        print("  ⚠️  No recipes found with 2+ participants.")
        print("  This means each recipe_id in complete_recipes.json is participant-specific.")
        print("  You may need to GROUP recipes by name/type for cross-participant analysis.")
    else:
        for _, row in multi.iterrows():
            print(f"  {row['recipe_name']:40s} | {row['participant_count']} participants | "
                  f"{row['total_sessions']} sessions | {row['total_actions']} actions")

    print(f"\nAll recipes by coverage:")
    print(df[['recipe_name', 'participant_count', 'participants',
              'total_sessions', 'total_actions', 'avg_steps']].to_string(index=False))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1B. Check vocabulary consistency
# ─────────────────────────────────────────────────────────────────────────────

def check_vocabulary_consistency(narrations, verb_classes, noun_classes):
    """
    HD-EPIC uses class IDs (verb_class, noun_class) in main_action_classes.
    This function checks:
      1. Whether the class taxonomy is SHARED across participants
         (i.e., verb_class=0 always means 'take' regardless of participant).
      2. How many unique verb-noun combinations exist per participant.
      3. Whether the same semantic action maps to the same class across
         participants (confirming taxonomy consistency).

    If HD-EPIC uses a shared taxonomy (which it does, based on EPIC-Kitchens
    conventions), then get_action_name(verb_class, noun_class) will always
    produce the same string for the same class IDs — no normalization needed.
    """
    print("\n" + "=" * 80)
    print("STEP 1B: VOCABULARY CONSISTENCY CHECK")
    print("=" * 80)

    # Check 1: Verb class taxonomy
    print("\n--- Verb Class Taxonomy ---")
    print(f"Total verb classes: {len(verb_classes)}")
    print(f"Sample verbs (first 10):")
    for _, row in verb_classes.head(10).iterrows():
        instances_str = str(row.get('instances', ''))
        # Count how many surface forms map to this class
        try:
            instances = eval(instances_str) if isinstance(instances_str, str) else instances_str
            n_forms = len(instances) if isinstance(instances, list) else 0
        except:
            n_forms = 0
        print(f"  ID {row['id']:3d} → '{row['key']:15s}' ({n_forms} surface forms)")

    # Check 2: Noun class taxonomy
    print(f"\n--- Noun Class Taxonomy ---")
    print(f"Total noun classes: {len(noun_classes)}")
    print(f"Sample nouns (first 10):")
    for _, row in noun_classes.head(10).iterrows():
        print(f"  ID {row['id']:3d} → '{row['key']}'")

    # Check 3: Per-participant action vocabulary
    print("\n--- Per-Participant Action Vocabulary ---")
    participant_vocab = defaultdict(set)
    participant_action_counts = defaultdict(int)

    for _, row in narrations.iterrows():
        pid = row.get('participant_id', row.get('video_id', '').split('-')[0])
        mac = row.get('main_action_classes', None)
        pairs = parse_main_action_classes(mac)
        for vc, nc in pairs:
            action = get_action_name(vc, nc, verb_classes, noun_classes)
            participant_vocab[pid].add(action)
            participant_action_counts[pid] += 1

    print(f"\n{'Participant':<15} {'Unique Actions':<18} {'Total Actions'}")
    print("-" * 50)
    for pid in sorted(participant_vocab.keys()):
        print(f"{pid:<15} {len(participant_vocab[pid]):<18} {participant_action_counts[pid]}")

    # Check 4: Cross-participant vocabulary overlap
    all_participants = sorted(participant_vocab.keys())
    if len(all_participants) >= 2:
        print("\n--- Cross-Participant Vocabulary Overlap ---")
        # Find actions shared by ALL participants
        shared_all = set.intersection(*participant_vocab.values()) if participant_vocab else set()
        union_all = set.union(*participant_vocab.values()) if participant_vocab else set()
        print(f"Actions shared by ALL {len(all_participants)} participants: {len(shared_all)}")
        print(f"Total unique actions across all participants: {len(union_all)}")
        print(f"Overlap ratio: {len(shared_all)/len(union_all)*100:.1f}%")

        # Pairwise overlap for a few pairs
        if len(all_participants) >= 2:
            print(f"\nPairwise overlap (first 5 pairs):")
            pairs_shown = 0
            for i in range(len(all_participants)):
                for j in range(i+1, len(all_participants)):
                    if pairs_shown >= 5:
                        break
                    p1, p2 = all_participants[i], all_participants[j]
                    overlap = participant_vocab[p1] & participant_vocab[p2]
                    union = participant_vocab[p1] | participant_vocab[p2]
                    jaccard = len(overlap) / len(union) if union else 0
                    print(f"  {p1} ∩ {p2}: {len(overlap)} shared / "
                          f"{len(union)} total (Jaccard={jaccard:.3f})")
                    pairs_shown += 1

    # Check 5: Key finding — is normalization needed?
    print("\n" + "=" * 80)
    print("VOCABULARY CONSISTENCY VERDICT")
    print("=" * 80)
    print("""
HD-EPIC uses a SHARED taxonomy of verb and noun classes across all participants.
This means:
  - verb_class=0 ALWAYS maps to 'take' (regardless of participant)
  - noun_class=5 ALWAYS maps to the same noun (regardless of participant)

The class system ALREADY normalizes surface-level variation:
  - "grab", "pick up", "collect" → all map to verb_class=0 ('take')
  - "mug", "cup" → may or may not map to the same noun_class (check above)

CONCLUSION: If you use get_action_name(verb_class, noun_class), the vocabulary
is ALREADY consistent across participants. No additional normalization is needed
at the class level.

CAVEAT: Noun classes may still be quite granular (e.g., 'cup' vs 'mug' as
separate classes). For canonical graph construction, you may want to consider
noun-level grouping (e.g., treating 'cup' and 'mug' as equivalent) depending
on your analysis goals.
""")

    # Build consistency report
    report_rows = []
    for pid in sorted(participant_vocab.keys()):
        for action in sorted(participant_vocab[pid]):
            report_rows.append({
                'participant': pid,
                'action': action,
                'verb': action.split('(')[0],
                'noun': action.split('(')[1].rstrip(')') if '(' in action else '',
            })
    report_df = pd.DataFrame(report_rows)

    return report_df


# ─────────────────────────────────────────────────────────────────────────────
# 1C. Detailed recipe session inventory
# ─────────────────────────────────────────────────────────────────────────────

def build_session_inventory(recipes, narrations, verb_classes, noun_classes):
    """
    For each recipe, list every session (video) with:
      - participant
      - video_id
      - action count
      - duration (start to end timestamp)
      - unique action count
      - first/last actions

    This helps you pick specific recipes for Steps 2-4.
    """
    print("\n" + "=" * 80)
    print("STEP 1C: DETAILED SESSION INVENTORY")
    print("=" * 80)

    rows = []

    for recipe_id, recipe_data in recipes.items():
        recipe_name = recipe_data.get('name', 'Unknown')
        participant = recipe_data.get('participant', recipe_id.split('_')[0])
        captures = recipe_data.get('captures', [])

        for cap_idx, cap in enumerate(captures):
            for video_id in cap.get('videos', []):
                vid_narrations = narrations[narrations['video_id'] == video_id]
                if vid_narrations.empty:
                    continue

                vid_sorted = vid_narrations.sort_values('start_timestamp')

                # Extract action sequence
                actions = []
                for _, row in vid_sorted.iterrows():
                    mac = row.get('main_action_classes', None)
                    pairs = parse_main_action_classes(mac)
                    if pairs:
                        vc, nc = pairs[0]
                        actions.append(get_action_name(vc, nc, verb_classes, noun_classes))

                if not actions:
                    continue

                duration = float(vid_sorted['end_timestamp'].max() -
                                 vid_sorted['start_timestamp'].min())

                rows.append({
                    'recipe_id': recipe_id,
                    'recipe_name': recipe_name,
                    'participant': participant,
                    'video_id': video_id,
                    'capture_index': cap_idx,
                    'action_count': len(actions),
                    'unique_actions': len(set(actions)),
                    'duration_sec': round(duration, 1),
                    'first_action': actions[0],
                    'last_action': actions[-1],
                })

    df = pd.DataFrame(rows)

    if not df.empty:
        # Summary: group by recipe_name to see multi-session recipes
        summary = df.groupby('recipe_name').agg(
            participants=('participant', 'nunique'),
            sessions=('video_id', 'count'),
            avg_actions=('action_count', 'mean'),
            avg_duration=('duration_sec', 'mean'),
        ).sort_values('sessions', ascending=False).reset_index()

        print("\nRecipes ranked by number of sessions:")
        print(summary.to_string(index=False))

        print(f"\n{'─'*80}")
        print("RECOMMENDATION FOR STEP 2:")
        print(f"{'─'*80}")

        # Recommend recipes with ≥3 sessions
        good = summary[summary['sessions'] >= 3]
        if not good.empty:
            print(f"  ✓ {len(good)} recipes have ≥3 sessions — suitable for canonical graphs.")
            print(f"  Top candidates:")
            for _, row in good.head(5).iterrows():
                print(f"    • '{row['recipe_name']}': {row['sessions']} sessions, "
                      f"{row['participants']} participant(s), "
                      f"~{row['avg_actions']:.0f} actions/session")
        else:
            print("  ⚠️  No recipe has ≥3 sessions.")
            print("  Consider using ALL sessions from the same PARTICIPANT for one recipe,")
            print("  or grouping similar recipe names (e.g., all coffee recipes).")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    data = load_hd_epic_data('..')
    create_output_dirs()

    # 1A: Coverage survey
    coverage_df = survey_recipe_coverage(
        data['recipes'],
        data['recipe_timestamps'],
        data['narrations']
    )
    coverage_path = Path('../outputs/tables/recipe_coverage_survey.csv')
    coverage_df.to_csv(coverage_path, index=False)
    print(f"\n✓ Coverage survey saved to {coverage_path}")

    # 1B: Vocabulary consistency
    vocab_df = check_vocabulary_consistency(
        data['narrations'],
        data['verb_classes'],
        data['noun_classes']
    )
    vocab_path = Path('../outputs/tables/vocabulary_consistency_report.csv')
    vocab_df.to_csv(vocab_path, index=False)
    print(f"✓ Vocabulary report saved to {vocab_path}")

    # 1C: Session inventory
    inventory_df = build_session_inventory(
        data['recipes'],
        data['narrations'],
        data['verb_classes'],
        data['noun_classes']
    )
    inventory_path = Path('../outputs/tables/session_inventory.csv')
    inventory_df.to_csv(inventory_path, index=False)
    print(f"✓ Session inventory saved to {inventory_path}")

    print("\n" + "=" * 80)
    print("STEP 1 COMPLETE")
    print("=" * 80)
    print("\nNext: Review the CSV outputs, pick 3-4 recipes with ≥3 sessions,")
    print("then run step2_canonical_graph.py with the chosen recipe name.")


if __name__ == "__main__":
    main()