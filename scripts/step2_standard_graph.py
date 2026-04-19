"""
Step 2: Canonical Recipe Graph Construction
=============================================
Goal: For a selected recipe, aggregate ALL participant sessions into a single
weighted graph that represents the "population norm" — the normative model.

This implements process mining concepts (van der Aalst, 2016) applied to
cooking activity sequences:
  - Each session = one "trace" in process mining terminology
  - The canonical graph = the "directly-follows graph" with frequency weights
  - Variance information = transition probabilities + timing distributions

Output:
  - outputs/graphs/canonical_graph_{recipe_name}.pkl
  - outputs/tables/canonical_transitions_{recipe_name}.csv
  - outputs/tables/canonical_node_stats_{recipe_name}.csv
  - outputs/figures/canonical_graph_{recipe_name}.png

Usage:
  python step2_canonical_graph.py "coffee"
  python step2_canonical_graph.py "P08_R01"
  (first arg = recipe name or recipe_id; matches case-insensitively)
"""

import sys
import pandas as pd
import numpy as np
import networkx as nx
import json
import pickle
from pathlib import Path
from collections import defaultdict, Counter
from utils import (load_hd_epic_data, create_output_dirs,
                   get_action_name, get_verb_name, get_noun_name)


# ─────────────────────────────────────────────────────────────────────────────
# 2A. Extract action traces from all sessions of a recipe
# ─────────────────────────────────────────────────────────────────────────────

def find_recipe_sessions(recipe_query, recipes, narrations):
    """
    Given a recipe query (name or ID), find all matching sessions.

    Returns:
      - matched_recipe_ids: list of recipe IDs (e.g., ['P01_R01', 'P08_R01'])
      - session_videos: dict mapping recipe_id → list of video_ids
      - recipe_meta: dict with name, steps, etc.
    """
    query = recipe_query.strip().lower()

    # Try exact recipe_id match first
    if query.upper() in recipes:
        matched = [query.upper()]
    else:
        # Match by name (case-insensitive, partial match)
        matched = [
            rid for rid, rdata in recipes.items()
            if query in rdata.get('name', '').lower()
        ]

    if not matched:
        print(f"⚠️  No recipe found matching '{recipe_query}'")
        print("Available recipe names:")
        names = sorted(set(r.get('name', '') for r in recipes.values()))
        for n in names[:20]:
            print(f"  • {n}")
        sys.exit(1)

    print(f"\nMatched {len(matched)} recipe definition(s) for query '{recipe_query}':")

    session_videos = {}
    for rid in matched:
        rdata = recipes[rid]
        vids = []
        for cap in rdata.get('captures', []):
            vids.extend(cap.get('videos', []))
        session_videos[rid] = vids
        participant = rdata.get('participant', rid.split('_')[0])
        print(f"  {rid} ({rdata.get('name')}) — participant {participant}, "
              f"{len(vids)} session(s)")

    # Use the first matched recipe's metadata as representative
    recipe_meta = {
        'name': recipes[matched[0]].get('name', 'Unknown'),
        'steps': recipes[matched[0]].get('steps', {}),
        'matched_ids': matched,
    }

    return matched, session_videos, recipe_meta


def extract_traces(session_videos, narrations, verb_classes, noun_classes):
    """
    Extract action traces (ordered action sequences) for each session.

    In process mining terms, each trace is one execution of the process.

    Returns:
      traces: list of dicts, each containing:
        - video_id
        - participant
        - actions: list of action strings (e.g., 'take(cup)')
        - timestamps: list of (start, end) tuples
        - durations: list of action durations
    """
    traces = []

    for recipe_id, video_ids in session_videos.items():
        participant = recipe_id.split('_')[0]

        for video_id in video_ids:
            vid_df = narrations[narrations['video_id'] == video_id].sort_values('start_timestamp')
            if vid_df.empty:
                continue

            actions = []
            timestamps = []
            durations = []

            for _, row in vid_df.iterrows():
                mac = row.get('main_action_classes', None)
                if mac and len(mac) > 0:
                    vc, nc = mac[0]
                    action = get_action_name(vc, nc, verb_classes, noun_classes)
                    start = float(row['start_timestamp'])
                    end = float(row['end_timestamp'])

                    actions.append(action)
                    timestamps.append((start, end))
                    durations.append(end - start)

            if actions:
                traces.append({
                    'video_id': video_id,
                    'participant': participant,
                    'recipe_id': recipe_id,
                    'actions': actions,
                    'timestamps': timestamps,
                    'durations': durations,
                })

    print(f"\nExtracted {len(traces)} traces across {len(session_videos)} recipe definition(s)")
    for t in traces:
        print(f"  {t['video_id']} ({t['participant']}): {len(t['actions'])} actions, "
              f"{t['timestamps'][-1][1] - t['timestamps'][0][0]:.1f}s duration")

    return traces


# ─────────────────────────────────────────────────────────────────────────────
# 2B. Build the canonical (directly-follows) graph
# ─────────────────────────────────────────────────────────────────────────────

def build_canonical_graph(traces):
    """
    Construct a weighted directed graph representing the normative model.

    Nodes = unique actions (verb(noun) pairs)
    Edges = directly-follows relationships with:
      - weight: number of traces containing this transition
      - frequency: total count of this transition across all traces
      - traces: which traces contain this transition
      - durations: list of inter-action spans (start_i+1 - start_i)
      - probability: P(next=B | current=A) for each edge A→B

    This is the "directly-follows graph" from process mining
    (van der Aalst, 2016, Chapter 6).
    """
    print("\n" + "=" * 80)
    print("BUILDING CANONICAL GRAPH")
    print("=" * 80)

    G = nx.DiGraph()

    # Collect statistics
    node_frequency = Counter()          # How many times each action appears total
    node_trace_count = Counter()        # In how many traces each action appears
    node_durations = defaultdict(list)  # Duration of each action instance
    edge_frequency = Counter()          # Total count of each transition
    edge_trace_set = defaultdict(set)   # Which traces contain each transition
    edge_spans = defaultdict(list)      # Time span between consecutive actions

    # Track first/last actions for START/END
    first_actions = Counter()
    last_actions = Counter()

    for trace_idx, trace in enumerate(traces):
        actions = trace['actions']
        timestamps = trace['timestamps']
        durations = trace['durations']
        trace_id = trace['video_id']

        # Track unique actions in this trace (for trace-level counts)
        seen_in_trace = set()

        for i, action in enumerate(actions):
            node_frequency[action] += 1
            node_durations[action].append(durations[i])
            if action not in seen_in_trace:
                node_trace_count[action] += 1
                seen_in_trace.add(action)

        # First and last
        if actions:
            first_actions[actions[0]] += 1
            last_actions[actions[-1]] += 1

        # Transitions
        for i in range(len(actions) - 1):
            a, b = actions[i], actions[i + 1]
            edge_frequency[(a, b)] += 1
            edge_trace_set[(a, b)].add(trace_id)

            # Inter-action span: time from start of action i to start of action i+1
            span = timestamps[i + 1][0] - timestamps[i][0]
            edge_spans[(a, b)].append(span)

    # Build the graph
    total_traces = len(traces)

    # Add nodes with statistics
    for action in node_frequency:
        G.add_node(action,
                   total_count=node_frequency[action],
                   trace_count=node_trace_count[action],
                   trace_fraction=node_trace_count[action] / total_traces,
                   avg_duration=float(np.mean(node_durations[action])),
                   std_duration=float(np.std(node_durations[action])),
                   median_duration=float(np.median(node_durations[action])),
                   )

    # Add edges with statistics
    # Also compute transition probabilities P(B | A)
    outgoing_totals = Counter()
    for (a, b), freq in edge_frequency.items():
        outgoing_totals[a] += freq

    for (a, b), freq in edge_frequency.items():
        spans = edge_spans[(a, b)]
        trace_ids = edge_trace_set[(a, b)]

        G.add_edge(a, b,
                   frequency=freq,
                   trace_count=len(trace_ids),
                   trace_fraction=len(trace_ids) / total_traces,
                   probability=freq / outgoing_totals[a],
                   avg_span=float(np.mean(spans)),
                   std_span=float(np.std(spans)),
                   median_span=float(np.median(spans)),
                   min_span=float(np.min(spans)),
                   max_span=float(np.max(spans)),
                   )

    # Add START and END sentinel nodes
    G.add_node('START', total_count=total_traces, trace_count=total_traces,
               trace_fraction=1.0, avg_duration=0, std_duration=0, median_duration=0)
    G.add_node('END', total_count=total_traces, trace_count=total_traces,
               trace_fraction=1.0, avg_duration=0, std_duration=0, median_duration=0)

    for action, count in first_actions.items():
        G.add_edge('START', action, frequency=count, trace_count=count,
                   trace_fraction=count/total_traces, probability=count/total_traces,
                   avg_span=0, std_span=0, median_span=0, min_span=0, max_span=0)
    for action, count in last_actions.items():
        G.add_edge(action, 'END', frequency=count, trace_count=count,
                   trace_fraction=count/total_traces, probability=count/total_traces,
                   avg_span=0, std_span=0, median_span=0, min_span=0, max_span=0)

    # Report
    print(f"\nCanonical graph statistics:")
    print(f"  Nodes (unique actions): {G.number_of_nodes() - 2} (+2 sentinels)")
    print(f"  Edges (transitions):    {G.number_of_edges()}")
    print(f"  Traces used:            {total_traces}")

    # Identify the "main path" — sequence of highest-probability transitions
    print(f"\n--- Most Probable Path (greedy) ---")
    path = ['START']
    current = 'START'
    for _ in range(50):  # safety limit
        successors = list(G.successors(current))
        if not successors or current == 'END':
            break
        # Pick highest probability successor
        best = max(successors, key=lambda s: G[current][s].get('probability', 0))
        path.append(best)
        current = best
    print(f"  {' → '.join(path)}")

    # Show top transitions by frequency
    print(f"\n--- Top 15 Transitions by Frequency ---")
    sorted_edges = sorted(G.edges(data=True),
                          key=lambda e: e[2].get('frequency', 0),
                          reverse=True)
    for u, v, d in sorted_edges[:15]:
        print(f"  {u:30s} → {v:30s}  "
              f"freq={d['frequency']:3d}  "
              f"P={d['probability']:.2f}  "
              f"span={d['avg_span']:.1f}±{d['std_span']:.1f}s")

    return G


# ─────────────────────────────────────────────────────────────────────────────
# 2C. Export canonical graph statistics as tables
# ─────────────────────────────────────────────────────────────────────────────

def export_canonical_tables(G, recipe_name, output_dir='../outputs/tables'):
    """Export node and edge statistics as CSV for inspection."""
    output_dir = Path(output_dir)
    safe_name = recipe_name.lower().replace(' ', '_').replace(',', '')

    # Node stats
    node_rows = []
    for node, data in G.nodes(data=True):
        if node in ('START', 'END'):
            continue
        node_rows.append({
            'action': node,
            'verb': node.split('(')[0],
            'noun': node.split('(')[1].rstrip(')') if '(' in node else '',
            'total_count': data.get('total_count', 0),
            'trace_count': data.get('trace_count', 0),
            'trace_fraction': round(data.get('trace_fraction', 0), 3),
            'avg_duration': round(data.get('avg_duration', 0), 2),
            'std_duration': round(data.get('std_duration', 0), 2),
            'median_duration': round(data.get('median_duration', 0), 2),
        })
    node_df = pd.DataFrame(node_rows).sort_values('total_count', ascending=False)
    node_path = output_dir / f'canonical_node_stats_{safe_name}.csv'
    node_df.to_csv(node_path, index=False)
    print(f"✓ Node stats saved to {node_path}")

    # Edge stats
    edge_rows = []
    for u, v, data in G.edges(data=True):
        edge_rows.append({
            'source': u,
            'target': v,
            'frequency': data.get('frequency', 0),
            'trace_count': data.get('trace_count', 0),
            'trace_fraction': round(data.get('trace_fraction', 0), 3),
            'probability': round(data.get('probability', 0), 3),
            'avg_span_sec': round(data.get('avg_span', 0), 2),
            'std_span_sec': round(data.get('std_span', 0), 2),
            'median_span_sec': round(data.get('median_span', 0), 2),
        })
    edge_df = pd.DataFrame(edge_rows).sort_values('frequency', ascending=False)
    edge_path = output_dir / f'canonical_transitions_{safe_name}.csv'
    edge_df.to_csv(edge_path, index=False)
    print(f"✓ Transition stats saved to {edge_path}")

    return node_df, edge_df


# ─────────────────────────────────────────────────────────────────────────────
# 2D. Visualize the canonical graph
# ─────────────────────────────────────────────────────────────────────────────

def visualize_canonical_graph(G, recipe_name, traces, output_path):
    """
    Visualize the canonical graph with:
      - Node size ∝ trace_fraction (how many sessions include this action)
      - Node color = verb category (same scheme as your v3)
      - Edge width ∝ frequency
      - Edge opacity ∝ trace_fraction (transitions seen in most sessions are bold)
      - Edge label = probability (for high-prob edges)
      - Layout: hierarchical left-to-right based on median sequence position
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    print("\n" + "=" * 80)
    print("VISUALIZING CANONICAL GRAPH")
    print("=" * 80)

    # --- Layout: median sequence position across traces ---
    node_seq_positions = defaultdict(list)
    for trace in traces:
        n = max(len(trace['actions']) - 1, 1)
        for i, action in enumerate(trace['actions']):
            node_seq_positions[action].append(i / n)

    N_COLS = 20
    X_SCALE, Y_SCALE = 3.0, 2.0
    col_buckets = defaultdict(list)
    for node in G.nodes():
        if node in ('START', 'END'):
            continue
        med = float(np.median(node_seq_positions.get(node, [0.5])))
        col = round(med * (N_COLS - 1))
        col_buckets[col].append(node)

    pos = {}
    for col, nodes_in_col in col_buckets.items():
        n = len(nodes_in_col)
        for i, node in enumerate(sorted(nodes_in_col)):
            y = (i - (n - 1) / 2.0) * Y_SCALE
            pos[node] = (col * X_SCALE, y)

    all_x = [p[0] for p in pos.values()] if pos else [0]
    pos['START'] = (min(all_x) - X_SCALE * 2, 0)
    pos['END'] = (max(all_x) + X_SCALE * 2, 0)

    # --- Verb color scheme ---
    VERB_COLORS = {
        'take': '#3B82F6', 'put': '#8B5CF6', 'open': '#06B6D4', 'close': '#06B6D4',
        'pour': '#F97316', 'scoop': '#F97316', 'mix': '#F97316',
        'press': '#EF4444', 'crush': '#EF4444',
        'turn-on': '#10B981', 'turn-off': '#10B981',
        'wait': '#6B7280', 'check': '#6B7280', 'carry': '#3B82F6',
        'move': '#3B82F6', 'slide': '#3B82F6', 'search': '#6B7280',
        'write': '#6B7280', 'adjust': '#6B7280', 'finish': '#10B981',
    }
    DEFAULT_COLOR = '#94A3B8'

    def get_color(node):
        if node in ('START', 'END'):
            return '#1F2937'
        verb = node.split('(')[0].lower().strip()
        return VERB_COLORS.get(verb, DEFAULT_COLOR)

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(30, 16))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # --- Edges ---
    max_freq = max((d.get('frequency', 1) for _, _, d in G.edges(data=True)), default=1)

    for u, v, data in G.edges(data=True):
        freq = data.get('frequency', 1)
        trace_frac = data.get('trace_fraction', 0)
        prob = data.get('probability', 0)

        lw = 0.8 + 4.0 * (freq / max_freq)
        alpha = max(0.15, min(0.9, trace_frac))

        # Determine if back-edge
        is_back = pos.get(v, (0,))[0] <= pos.get(u, (0,))[0] and u not in ('START',) and v not in ('END',)

        if u == v:
            # Self-loop
            x, y = pos[u]
            loop = mpatches.FancyArrowPatch(
                posA=(x - 0.3, y + 0.6), posB=(x + 0.3, y + 0.6),
                arrowstyle=mpatches.ArrowStyle('Simple', head_width=6, head_length=5),
                connectionstyle='arc3,rad=-1.2',
                color='#64748B', linewidth=lw, alpha=alpha, linestyle='dashed', zorder=2)
            ax.add_patch(loop)
        elif is_back:
            rad = 0.4 + 0.3 * min(abs(pos[u][0] - pos[v][0]) / max(max(all_x) - min(all_x), 1), 0.8)
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=lw,
                                   edge_color='#64748B', alpha=alpha, arrows=True,
                                   arrowsize=14, arrowstyle='->', style='dashed',
                                   connectionstyle=f'arc3,rad={rad}', ax=ax,
                                   min_source_margin=20, min_target_margin=20)
        else:
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=lw,
                                   edge_color='#475569', alpha=alpha, arrows=True,
                                   arrowsize=16, arrowstyle='->',
                                   connectionstyle='arc3,rad=-0.05', ax=ax,
                                   min_source_margin=20, min_target_margin=20)

        # Label high-probability edges
        if prob >= 0.5 and u not in ('START',) and v not in ('END',):
            mid_x = (pos[u][0] + pos[v][0]) / 2
            mid_y = (pos[u][1] + pos[v][1]) / 2 + 0.3
            ax.text(mid_x, mid_y, f'{prob:.0%}', fontsize=6, color='#334155',
                    ha='center', va='center', alpha=0.7,
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # --- Nodes ---
    max_trace_count = max((d.get('trace_count', 1) for _, d in G.nodes(data=True)), default=1)

    for node, data in G.nodes(data=True):
        x, y = pos[node]
        tc = data.get('trace_count', 1)

        if node in ('START', 'END'):
            circle = plt.Circle((x, y), 0.55, color='#1F2937', zorder=4)
            ax.add_patch(circle)
            ax.text(x, y, node, fontsize=9, fontweight='bold', color='white',
                    ha='center', va='center', zorder=5)
        else:
            size_r = 0.3 + 0.5 * (tc / max_trace_count)
            color = get_color(node)

            # Outline thickness indicates trace_fraction
            trace_frac = data.get('trace_fraction', 0)
            edge_lw = 1.0 + 2.5 * trace_frac

            circle = plt.Circle((x, y), size_r, color=color,
                                ec='#1e293b', linewidth=edge_lw, zorder=4)
            ax.add_patch(circle)

            verb = node.split('(')[0]
            noun_part = node[len(verb):]
            ax.text(x, y, verb, fontsize=7, fontweight='bold', color='white',
                    ha='center', va='center', zorder=5,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground=color)])
            ax.text(x, y - size_r - 0.18, noun_part, fontsize=6.5, color='#334155',
                    ha='center', va='top', zorder=5,
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

            # Show trace fraction below noun
            ax.text(x, y - size_r - 0.55,
                    f'{trace_frac:.0%} sessions',
                    fontsize=5.5, color='#64748B',
                    ha='center', va='top', zorder=5,
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # --- Legend ---
    edge_legend = [
        Line2D([0], [0], color='#475569', lw=2.5, label='Forward transition'),
        Line2D([0], [0], color='#64748B', lw=2.5, linestyle='dashed', label='Back-edge / cycle'),
        Line2D([0], [0], color='none', label='Edge width ∝ frequency'),
        Line2D([0], [0], color='none', label='Edge opacity ∝ session coverage'),
    ]
    node_legend = [
        Patch(color='#3B82F6', label='Take / carry / move'),
        Patch(color='#8B5CF6', label='Put / place'),
        Patch(color='#F97316', label='Pour / scoop / mix'),
        Patch(color='#EF4444', label='Press / crush'),
        Patch(color='#06B6D4', label='Open / close'),
        Patch(color='#10B981', label='Machine ops / finish'),
        Patch(color='#6B7280', label='Wait / check / search'),
        Patch(color='#94A3B8', label='Other'),
    ]

    leg1 = ax.legend(handles=edge_legend, loc='upper left', fontsize=9,
                     title='Edge encoding', title_fontsize=10,
                     framealpha=0.95, edgecolor='#CBD5E1')
    ax.add_artist(leg1)
    ax.legend(handles=node_legend, loc='lower left', fontsize=9,
              title='Node color = verb category', title_fontsize=10,
              framealpha=0.95, edgecolor='#CBD5E1')

    # --- Title ---
    n_traces = len(traces)
    n_participants = len(set(t['participant'] for t in traces))
    ax.set_title(
        f'Canonical Recipe Graph: {recipe_name}\n'
        f'{G.number_of_nodes() - 2} action states  ·  '
        f'{G.number_of_edges()} transitions  ·  '
        f'{n_traces} sessions from {n_participants} participant(s)\n'
        f'Node size & border ∝ session coverage  |  Edge width ∝ frequency  |  Labels = P(transition)',
        fontsize=14, fontweight='bold', pad=16)
    ax.axis('off')

    all_positions = list(pos.values())
    all_x = [p[0] for p in all_positions]
    all_y = [p[1] for p in all_positions]
    ax.set_xlim(min(all_x) - 3, max(all_x) + 3)
    ax.set_ylim(min(all_y) - 3, max(all_y) + 3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✓ Canonical graph visualization saved to {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python step2_canonical_graph.py <recipe_name_or_id>")
        print("Example: python step2_canonical_graph.py coffee")
        print("         python step2_canonical_graph.py P08_R01")
        sys.exit(1)

    recipe_query = sys.argv[1]

    data = load_hd_epic_data('..')
    create_output_dirs()

    # Find all sessions for this recipe
    matched_ids, session_videos, recipe_meta = find_recipe_sessions(
        recipe_query, data['recipes'], data['narrations']
    )
    recipe_name = recipe_meta['name']
    safe_name = recipe_name.lower().replace(' ', '_').replace(',', '')

    # Extract traces
    traces = extract_traces(
        session_videos, data['narrations'],
        data['verb_classes'], data['noun_classes']
    )

    if not traces:
        print("ERROR: No traces extracted. Check video_id matching.")
        sys.exit(1)

    # Build canonical graph
    G = build_canonical_graph(traces)

    # Export tables
    export_canonical_tables(G, recipe_name)

    # Save graph object
    graph_path = Path(f'../outputs/graphs/canonical_graph_{safe_name}.pkl')
    with open(graph_path, 'wb') as f:
        pickle.dump({'graph': G, 'traces': traces, 'meta': recipe_meta}, f)
    print(f"✓ Graph data saved to {graph_path}")

    # Visualize
    fig_path = f'../outputs/figures/canonical_graph_{safe_name}.png'
    visualize_canonical_graph(G, recipe_name, traces, fig_path)

    print("\n" + "=" * 80)
    print("STEP 2 COMPLETE")
    print("=" * 80)
    print(f"\nCanonical graph for '{recipe_name}' built from {len(traces)} traces.")
    print("Next: Run step3_deviation_detection.py to detect individual deviations.")


if __name__ == "__main__":
    main()