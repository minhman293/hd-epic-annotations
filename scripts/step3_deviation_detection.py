"""
Step 3: Deviation Detection
=============================
Goal: Using the canonical graph as baseline, detect and classify deviations
in individual sessions. Four deviation types, ordered by intervention relevance:

  1. OMISSIONS — Critical steps present in the canonical model but absent
     from a session. Grounded in Schwartz (2006) omission error taxonomy.

  2. ORDERING DEVIATIONS — Actions performed in a different sequence than
     the canonical path. Measured via Damerau-Levenshtein edit distance.

  3. OSCILLATIONS — A→B→A patterns indicating searching or confusion.
     Grounded in Kirsh & Maglio (1994) epistemic vs. pragmatic action.

  4. RARE TRANSITIONS — Transitions that appear in <N% of sessions in the
     canonical model, indicating unusual/non-normative behavior.

Output:
  - outputs/tables/deviation_report_{recipe}.csv
  - outputs/tables/deviation_summary_{recipe}.csv
  - Console report with per-session breakdown

Usage:
  python step3_deviation_detection.py coffee
  python step3_deviation_detection.py P08_R01
"""

import sys
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from collections import Counter, defaultdict
from utils import load_hd_epic_data, create_output_dirs, get_action_name


# ─────────────────────────────────────────────────────────────────────────────
# 3A. Load canonical graph from Step 2
# ─────────────────────────────────────────────────────────────────────────────

def load_canonical_data(recipe_query, outputs_dir='../outputs'):
    """Load the canonical graph and traces saved by Step 2."""
    outputs_path = Path(outputs_dir)
    query = recipe_query.strip().lower().replace(' ', '_').replace(',', '')

    # Find matching canonical graph file
    candidates = list(outputs_path.glob(f'graphs/canonical_graph_*{query}*.pkl'))
    if not candidates:
        # Try all files
        candidates = list(outputs_path.glob('graphs/canonical_graph_*.pkl'))
        print(f"Available canonical graphs:")
        for c in candidates:
            print(f"  {c.name}")
        raise FileNotFoundError(
            f"No canonical graph found for '{recipe_query}'. "
            "Run step2_canonical_graph.py first."
        )

    graph_path = candidates[0]
    print(f"Loading canonical data from: {graph_path.name}")

    with open(graph_path, 'rb') as f:
        data = pickle.load(f)

    return data['graph'], data['traces'], data['meta']


# ─────────────────────────────────────────────────────────────────────────────
# 3B. Extract the canonical action set and canonical sequence
# ─────────────────────────────────────────────────────────────────────────────

def extract_canonical_reference(G, traces):
    """
    Derive the canonical reference from the graph:
      - canonical_actions: set of actions that appear in ≥50% of traces
      - canonical_sequence: the most-probable path through the graph
      - critical_actions: actions in ≥80% of traces (high intervention priority if omitted)
    """
    total_traces = len(traces)

    # Action coverage
    canonical_actions = set()
    critical_actions = set()

    for node, data in G.nodes(data=True):
        if node in ('START', 'END'):
            continue
        frac = data.get('trace_fraction', 0)
        if frac >= 0.5:
            canonical_actions.add(node)
        if frac >= 0.8:
            critical_actions.add(node)

    # Most-probable path (greedy)
    canonical_sequence = []
    current = 'START'
    visited = set()
    for _ in range(100):
        successors = [s for s in G.successors(current) if s not in visited or s == 'END']
        if not successors or current == 'END':
            break
        best = max(successors, key=lambda s: G[current][s].get('probability', 0))
        if best == 'END':
            break
        canonical_sequence.append(best)
        visited.add(best)
        current = best

    print(f"\nCanonical reference:")
    print(f"  Actions in ≥50% of sessions (canonical set): {len(canonical_actions)}")
    print(f"  Actions in ≥80% of sessions (critical set):  {len(critical_actions)}")
    print(f"  Most-probable path length: {len(canonical_sequence)}")
    print(f"  Path: {' → '.join(canonical_sequence[:15])}{'...' if len(canonical_sequence) > 15 else ''}")

    return canonical_actions, critical_actions, canonical_sequence


# ─────────────────────────────────────────────────────────────────────────────
# 3C. Deviation Type 1: Omission Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_omissions(trace, canonical_actions, critical_actions):
    """
    Find actions in the canonical set that are MISSING from this trace.

    Returns list of dicts with:
      - action: the omitted action
      - severity: 'critical' (in ≥80% of sessions) or 'moderate' (in ≥50%)
      - canonical_coverage: fraction of sessions that include this action

    Reference: Schwartz (2006) — omission is the most common error type
    in action disorganization.
    """
    session_actions = set(trace['actions'])
    omissions = []

    for action in canonical_actions:
        if action not in session_actions:
            severity = 'critical' if action in critical_actions else 'moderate'
            omissions.append({
                'deviation_type': 'omission',
                'action': action,
                'severity': severity,
                'detail': f'Missing from session; present in ≥{"80" if severity == "critical" else "50"}% of sessions',
            })

    return omissions


# ─────────────────────────────────────────────────────────────────────────────
# 3D. Deviation Type 2: Ordering Deviation
# ─────────────────────────────────────────────────────────────────────────────

def damerau_levenshtein_distance(s1, s2):
    """
    Compute the Damerau-Levenshtein distance between two sequences.
    This accounts for insertions, deletions, substitutions, AND transpositions.

    Optimal for detecting ordering swaps in procedural tasks.
    """
    len1, len2 = len(s1), len(s2)
    # Create distance matrix
    d = {}
    for i in range(-1, len1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, len2 + 1):
        d[(-1, j)] = j + 1

    for i in range(len1):
        for j in range(len2):
            cost = 0 if s1[i] == s2[j] else 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,       # deletion
                d[(i, j - 1)] + 1,        # insertion
                d[(i - 1, j - 1)] + cost, # substitution
            )
            # Transposition
            if i > 0 and j > 0 and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[(i - 2, j - 2)] + cost)

    return d[(len1 - 1, len2 - 1)]


def detect_ordering_deviation(trace, canonical_sequence):
    """
    Compare the trace's action sequence to the canonical (most-probable) sequence.

    Returns:
      - edit_distance: Damerau-Levenshtein distance
      - normalized_distance: distance / max(len(trace), len(canonical))
      - specific_swaps: pairs of actions that appear in swapped order
    """
    trace_actions = trace['actions']

    # Filter both sequences to their intersection for meaningful comparison
    canonical_set = set(canonical_sequence)
    trace_set = set(trace_actions)
    common = canonical_set & trace_set

    if len(common) < 2:
        return {
            'deviation_type': 'ordering',
            'edit_distance': 0,
            'normalized_distance': 0.0,
            'common_actions': len(common),
            'swapped_pairs': [],
            'detail': 'Too few common actions for meaningful comparison',
        }

    # Extract subsequences of common actions only (preserving order)
    canon_sub = [a for a in canonical_sequence if a in common]
    trace_sub = []
    seen = set()
    for a in trace_actions:
        if a in common and a not in seen:
            trace_sub.append(a)
            seen.add(a)

    edit_dist = damerau_levenshtein_distance(canon_sub, trace_sub)
    max_len = max(len(canon_sub), len(trace_sub), 1)
    norm_dist = edit_dist / max_len

    # Find specific swapped pairs
    # A pair (A, B) is "swapped" if A comes before B in canonical but after B in trace
    swapped_pairs = []
    canon_order = {a: i for i, a in enumerate(canon_sub)}
    trace_order = {a: i for i, a in enumerate(trace_sub)}

    for a in common:
        for b in common:
            if a >= b:
                continue
            if a in canon_order and b in canon_order and a in trace_order and b in trace_order:
                canon_before = canon_order[a] < canon_order[b]
                trace_before = trace_order[a] < trace_order[b]
                if canon_before != trace_before:
                    swapped_pairs.append((a, b))

    return {
        'deviation_type': 'ordering',
        'edit_distance': edit_dist,
        'normalized_distance': round(norm_dist, 3),
        'common_actions': len(common),
        'swapped_pairs': swapped_pairs,
        'detail': f'Edit distance={edit_dist}, {len(swapped_pairs)} swapped pairs',
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3E. Deviation Type 3: Oscillation Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_oscillations(trace):
    """
    Detect A→B→A patterns in the action sequence.

    Per Kirsh & Maglio (1994), not all oscillation is confusion —
    some is "epistemic action" (thinking with the environment).
    We flag all oscillations and let the human designer judge context.

    Returns list of oscillation instances with context.
    """
    actions = trace['actions']
    timestamps = trace['timestamps']
    oscillations = []

    for i in range(len(actions) - 2):
        if actions[i] == actions[i + 2] and actions[i] != actions[i + 1]:
            # A→B→A pattern found
            a, b = actions[i], actions[i + 1]

            # Compute time span of the oscillation
            span = timestamps[i + 2][1] - timestamps[i][0]

            oscillations.append({
                'deviation_type': 'oscillation',
                'action_a': a,
                'action_b': b,
                'position_in_sequence': i,
                'span_seconds': round(span, 2),
                'detail': f'{a} → {b} → {a} at position {i} (span={span:.1f}s)',
            })

    return oscillations


# ─────────────────────────────────────────────────────────────────────────────
# 3F. Deviation Type 4: Rare Transition Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_rare_transitions(trace, G, threshold=0.1):
    """
    Flag transitions in this trace that appear in fewer than `threshold`
    fraction of sessions in the canonical graph.

    threshold=0.1 means "transitions seen in fewer than 10% of sessions."

    Also flags transitions NOT present in the canonical graph at all (novel).
    """
    actions = trace['actions']
    rare = []

    for i in range(len(actions) - 1):
        a, b = actions[i], actions[i + 1]

        if G.has_edge(a, b):
            trace_frac = G[a][b].get('trace_fraction', 0)
            if trace_frac < threshold:
                rare.append({
                    'deviation_type': 'rare_transition',
                    'source': a,
                    'target': b,
                    'position': i,
                    'trace_fraction': round(trace_frac, 3),
                    'detail': f'{a} → {b} seen in only {trace_frac:.0%} of sessions',
                })
        else:
            # Novel transition — not in canonical graph at all
            rare.append({
                'deviation_type': 'novel_transition',
                'source': a,
                'target': b,
                'position': i,
                'trace_fraction': 0.0,
                'detail': f'{a} → {b} NEVER seen in canonical model',
            })

    return rare


# ─────────────────────────────────────────────────────────────────────────────
# 3G. Aggregate all deviations for one trace
# ─────────────────────────────────────────────────────────────────────────────

def analyze_trace(trace, G, canonical_actions, critical_actions, canonical_sequence,
                  rare_threshold=0.1):
    """Run all four deviation detectors on a single trace."""

    omissions = detect_omissions(trace, canonical_actions, critical_actions)
    ordering = detect_ordering_deviation(trace, canonical_sequence)
    oscillations = detect_oscillations(trace)
    rare = detect_rare_transitions(trace, G, threshold=rare_threshold)

    return {
        'video_id': trace['video_id'],
        'participant': trace['participant'],
        'action_count': len(trace['actions']),
        'unique_actions': len(set(trace['actions'])),
        'omissions': omissions,
        'ordering': ordering,
        'oscillations': oscillations,
        'rare_transitions': rare,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3H. Generate reports
# ─────────────────────────────────────────────────────────────────────────────

def generate_reports(all_results, recipe_name, output_dir='../outputs/tables'):
    """Generate CSV reports from deviation analysis."""
    output_dir = Path(output_dir)
    safe_name = recipe_name.lower().replace(' ', '_').replace(',', '')

    # 1. Detailed deviation report (one row per deviation instance)
    detail_rows = []
    for result in all_results:
        vid = result['video_id']
        pid = result['participant']

        for om in result['omissions']:
            detail_rows.append({
                'video_id': vid, 'participant': pid,
                'deviation_type': 'omission',
                'severity': om['severity'],
                'detail': om['detail'],
                'action': om['action'],
            })

        ord_dev = result['ordering']
        if ord_dev['edit_distance'] > 0:
            detail_rows.append({
                'video_id': vid, 'participant': pid,
                'deviation_type': 'ordering',
                'severity': 'high' if ord_dev['normalized_distance'] > 0.5 else
                            'moderate' if ord_dev['normalized_distance'] > 0.2 else 'low',
                'detail': ord_dev['detail'],
                'action': f"edit_dist={ord_dev['edit_distance']}",
            })
            for a, b in ord_dev['swapped_pairs']:
                detail_rows.append({
                    'video_id': vid, 'participant': pid,
                    'deviation_type': 'ordering_swap',
                    'severity': 'moderate',
                    'detail': f'{a} and {b} in reversed order vs canonical',
                    'action': f'{a} <-> {b}',
                })

        for osc in result['oscillations']:
            detail_rows.append({
                'video_id': vid, 'participant': pid,
                'deviation_type': 'oscillation',
                'severity': 'moderate',
                'detail': osc['detail'],
                'action': f"{osc['action_a']} ↔ {osc['action_b']}",
            })

        for rt in result['rare_transitions']:
            detail_rows.append({
                'video_id': vid, 'participant': pid,
                'deviation_type': rt['deviation_type'],
                'severity': 'high' if rt['trace_fraction'] == 0 else 'low',
                'detail': rt['detail'],
                'action': f"{rt['source']} → {rt['target']}",
            })

    detail_df = pd.DataFrame(detail_rows)
    detail_path = output_dir / f'deviation_report_{safe_name}.csv'
    detail_df.to_csv(detail_path, index=False)
    print(f"\n✓ Detailed deviation report saved to {detail_path}")
    print(f"  Total deviations found: {len(detail_df)}")

    # 2. Summary per session
    summary_rows = []
    for result in all_results:
        ord_dev = result['ordering']
        summary_rows.append({
            'video_id': result['video_id'],
            'participant': result['participant'],
            'action_count': result['action_count'],
            'unique_actions': result['unique_actions'],
            'n_omissions_critical': sum(1 for o in result['omissions'] if o['severity'] == 'critical'),
            'n_omissions_moderate': sum(1 for o in result['omissions'] if o['severity'] == 'moderate'),
            'edit_distance': ord_dev['edit_distance'],
            'normalized_edit_distance': ord_dev['normalized_distance'],
            'n_swapped_pairs': len(ord_dev['swapped_pairs']),
            'n_oscillations': len(result['oscillations']),
            'n_rare_transitions': sum(1 for r in result['rare_transitions']
                                       if r['deviation_type'] == 'rare_transition'),
            'n_novel_transitions': sum(1 for r in result['rare_transitions']
                                       if r['deviation_type'] == 'novel_transition'),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / f'deviation_summary_{safe_name}.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Summary saved to {summary_path}")

    # Console summary
    print(f"\n{'═'*80}")
    print(f"DEVIATION SUMMARY PER SESSION")
    print(f"{'═'*80}")
    for _, row in summary_df.iterrows():
        total_devs = (row['n_omissions_critical'] + row['n_omissions_moderate'] +
                      (1 if row['edit_distance'] > 0 else 0) +
                      row['n_oscillations'] + row['n_rare_transitions'] +
                      row['n_novel_transitions'])

        flag = '🔴' if row['n_omissions_critical'] > 0 or row['n_novel_transitions'] > 2 else \
               '🟡' if total_devs > 3 else '🟢'

        print(f"\n{flag} {row['video_id']} ({row['participant']})")
        print(f"   Actions: {row['action_count']} total, {row['unique_actions']} unique")
        print(f"   Omissions: {row['n_omissions_critical']} critical, {row['n_omissions_moderate']} moderate")
        print(f"   Ordering: edit_dist={row['edit_distance']}, {row['n_swapped_pairs']} swaps")
        print(f"   Oscillations: {row['n_oscillations']}")
        print(f"   Rare/novel transitions: {row['n_rare_transitions']} rare, {row['n_novel_transitions']} novel")

    # Cross-session patterns
    if len(summary_df) > 1:
        print(f"\n{'─'*80}")
        print("CROSS-SESSION PATTERNS")
        print(f"{'─'*80}")
        print(f"  Avg omissions (critical): {summary_df['n_omissions_critical'].mean():.1f}")
        print(f"  Avg oscillations: {summary_df['n_oscillations'].mean():.1f}")
        print(f"  Avg edit distance: {summary_df['edit_distance'].mean():.1f}")

        # Find most commonly omitted actions
        if not detail_df.empty:
            omission_counts = detail_df[detail_df['deviation_type'] == 'omission']['action'].value_counts()
            if not omission_counts.empty:
                print(f"\n  Most commonly omitted actions:")
                for action, count in omission_counts.head(5).items():
                    print(f"    • {action}: omitted in {count}/{len(all_results)} sessions")

    return detail_df, summary_df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python step3_deviation_detection.py <recipe_name_or_id>")
        sys.exit(1)

    recipe_query = sys.argv[1]
    create_output_dirs()

    # Load canonical graph from Step 2
    G, traces, meta = load_canonical_data(recipe_query)
    recipe_name = meta.get('name', recipe_query)

    print(f"\nAnalyzing deviations for '{recipe_name}'")
    print(f"Canonical graph: {G.number_of_nodes()-2} actions, {G.number_of_edges()} transitions")
    print(f"Traces to analyze: {len(traces)}")

    # Extract canonical reference
    canonical_actions, critical_actions, canonical_sequence = extract_canonical_reference(G, traces)

    # Analyze each trace
    all_results = []
    for trace in traces:
        result = analyze_trace(
            trace, G, canonical_actions, critical_actions, canonical_sequence,
            rare_threshold=0.15  # transitions in <15% of sessions
        )
        all_results.append(result)

    # Generate reports
    detail_df, summary_df = generate_reports(all_results, recipe_name)

    # Save full results as pickle for Step 4
    results_path = Path(f'../outputs/graphs/deviation_results_{recipe_name.lower().replace(" ", "_")}.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump({
            'results': all_results,
            'canonical_actions': canonical_actions,
            'critical_actions': critical_actions,
            'canonical_sequence': canonical_sequence,
            'recipe_name': recipe_name,
        }, f)
    print(f"\n✓ Full results saved to {results_path}")

    print("\n" + "=" * 80)
    print("STEP 3 COMPLETE")
    print("=" * 80)
    print("\nNext: Run step4_deviation_visualization.py to create the")
    print("intervention-oriented visualization overlay.")


if __name__ == "__main__":
    main()