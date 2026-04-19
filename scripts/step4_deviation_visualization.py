"""
Step 4: Deviation-Overlay Visualization
=========================================
Goal: Create a visualization that helps a human designer inspect the canonical
model, explore deviations across participants, and identify where intervention
rules would be warranted.

Design principles (grounded in Munzner 2014, Wongsuphasawat et al. CHI 2012):
  - The canonical path is the visual SPINE (bold, central, horizontal)
  - Deviation indicators are layered on top (color-coded by type)
  - "Intervention hotspots" = nodes/edges where many sessions deviate
  - Interactive-ready: outputs data for potential web-based tool (Step 5+)

Outputs:
  - outputs/figures/deviation_overlay_{recipe}.png  — static publication figure
  - outputs/tables/intervention_hotspots_{recipe}.csv — ranked list of hotspots

Usage:
  python step4_deviation_visualization.py coffee
    python step4_deviation_visualization.py coffee --level abstracted
    python step4_deviation_visualization.py coffee --level action
"""

import sys
import pandas as pd
import numpy as np
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyBboxPatch
from pathlib import Path
from collections import defaultdict, Counter
from utils import create_output_dirs


def normalize_safe_name(text):
    return str(text).strip().lower().replace(' ', '_').replace(',', '')


def parse_cli_args(argv):
    if not argv:
        return None, 'abstracted'

    recipe_query = None
    level = 'abstracted'
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == '--level' and i + 1 < len(argv):
            level = argv[i + 1].strip().lower()
            i += 2
            continue
        if arg.startswith('--level='):
            level = arg.split('=', 1)[1].strip().lower()
            i += 1
            continue
        if recipe_query is None and not arg.startswith('--'):
            recipe_query = arg
        i += 1

    if level not in ('abstracted', 'action'):
        raise ValueError("--level must be either 'abstracted' or 'action'")

    return recipe_query, level


# ─────────────────────────────────────────────────────────────────────────────
# 4A. Load data from Steps 2 and 3
# ─────────────────────────────────────────────────────────────────────────────

def load_step2_step3_data(recipe_query, level='abstracted', outputs_dir='../outputs'):
    """Load canonical graph (Step 2) and deviation results (Step 3)."""
    outputs_path = Path(outputs_dir)
    query = normalize_safe_name(recipe_query)

    # Canonical graph
    canon_candidates = []
    preferred_canon = outputs_path / f'graphs/canonical_graph_{query}_{level}.pkl'
    if preferred_canon.exists():
        canon_candidates = [preferred_canon]
    else:
        canon_candidates = list(outputs_path.glob(f'graphs/canonical_graph_*{query}*.pkl'))

    filtered_canon = []
    for c in canon_candidates:
        try:
            with open(c, 'rb') as f:
                d = pickle.load(f)
            c_level = d.get('meta', {}).get('abstraction_level')
            if c_level is None and c.name.endswith(f'_{level}.pkl'):
                filtered_canon.append(c)
            elif c_level == level:
                filtered_canon.append(c)
        except Exception:
            continue
    if filtered_canon:
        canon_candidates = filtered_canon

    if not canon_candidates:
        raise FileNotFoundError(f"No canonical graph for '{recipe_query}' at level '{level}'. Run step2 first.")
    with open(canon_candidates[0], 'rb') as f:
        canon_data = pickle.load(f)

    # Deviation results
    dev_candidates = []
    preferred_dev = outputs_path / f'graphs/deviation_results_{query}_{level}.pkl'
    if preferred_dev.exists():
        dev_candidates = [preferred_dev]
    else:
        dev_candidates = list(outputs_path.glob(f'graphs/deviation_results_*{query}*.pkl'))

    filtered_dev = []
    for c in dev_candidates:
        try:
            with open(c, 'rb') as f:
                d = pickle.load(f)
            d_level = d.get('abstraction_level')
            if d_level is None and c.name.endswith(f'_{level}.pkl'):
                filtered_dev.append(c)
            elif d_level == level:
                filtered_dev.append(c)
        except Exception:
            continue
    if filtered_dev:
        dev_candidates = filtered_dev

    if not dev_candidates:
        raise FileNotFoundError(f"No deviation results for '{recipe_query}' at level '{level}'. Run step3 first.")
    with open(dev_candidates[0], 'rb') as f:
        dev_data = pickle.load(f)

    canon_data.setdefault('meta', {})['abstraction_level'] = canon_data.get('meta', {}).get('abstraction_level', level)
    dev_data['abstraction_level'] = dev_data.get('abstraction_level', level)

    return canon_data, dev_data


# ─────────────────────────────────────────────────────────────────────────────
# 4B. Compute intervention hotspots
# ─────────────────────────────────────────────────────────────────────────────

def compute_intervention_hotspots(G, dev_results, canonical_sequence, traces):
    """
    For each node and edge in the canonical graph, compute an
    "intervention score" based on how many and what type of deviations
    occur at that point.

    Scoring (weights reflect intervention priority):
      - Critical omission at node:  +3 per session
      - Moderate omission at node:  +1 per session
      - Oscillation involving node: +2 per session
      - Ordering swap involving node: +1 per pair per session
      - Rare/novel transition at edge: +2 per session

    Returns:
      - node_scores: dict {node: float}
      - edge_scores: dict {(u,v): float}
      - hotspot_df: DataFrame ranked by score
    """
    node_scores = defaultdict(float)
    node_deviation_details = defaultdict(lambda: defaultdict(int))
    edge_scores = defaultdict(float)

    total_sessions = len(dev_results)

    for result in dev_results:
        # Omissions
        for om in result['omissions']:
            action = om['action']
            if om['severity'] == 'critical':
                node_scores[action] += 3.0
                node_deviation_details[action]['omission_critical'] += 1
            else:
                node_scores[action] += 1.0
                node_deviation_details[action]['omission_moderate'] += 1

        # Oscillations
        for osc in result['oscillations']:
            a = osc['action_a']
            b = osc['action_b']
            node_scores[a] += 2.0
            node_scores[b] += 1.0
            node_deviation_details[a]['oscillation'] += 1
            node_deviation_details[b]['oscillation_target'] += 1

        # Ordering swaps
        for a, b in result['ordering'].get('swapped_pairs', []):
            node_scores[a] += 1.0
            node_scores[b] += 1.0
            node_deviation_details[a]['ordering_swap'] += 1
            node_deviation_details[b]['ordering_swap'] += 1

        # Rare/novel transitions
        for rt in result['rare_transitions']:
            src, tgt = rt['source'], rt['target']
            edge_key = (src, tgt)
            if rt['deviation_type'] == 'novel_transition':
                edge_scores[edge_key] += 3.0
            else:
                edge_scores[edge_key] += 1.5
            node_scores[src] += 0.5
            node_scores[tgt] += 0.5
            node_deviation_details[src]['rare_transition_source'] += 1
            node_deviation_details[tgt]['rare_transition_target'] += 1

    # Normalize scores by number of sessions
    for node in node_scores:
        node_scores[node] /= total_sessions
    for edge in edge_scores:
        edge_scores[edge] /= total_sessions

    # Build ranked hotspot table
    hotspot_rows = []
    for node, score in sorted(node_scores.items(), key=lambda x: -x[1]):
        details = node_deviation_details[node]
        hotspot_rows.append({
            'action': node,
            'intervention_score': round(score, 2),
            'omission_critical': details.get('omission_critical', 0),
            'omission_moderate': details.get('omission_moderate', 0),
            'oscillation': details.get('oscillation', 0),
            'ordering_swap': details.get('ordering_swap', 0),
            'rare_transition': details.get('rare_transition_source', 0) +
                               details.get('rare_transition_target', 0),
            'in_canonical_path': node in canonical_sequence,
        })

    hotspot_df = pd.DataFrame(hotspot_rows)

    print(f"\n{'═'*80}")
    print("INTERVENTION HOTSPOTS (ranked by score)")
    print(f"{'═'*80}")
    if not hotspot_df.empty:
        print(hotspot_df.head(15).to_string(index=False))
    else:
        print("  No hotspots found (all sessions follow canonical path)")

    return dict(node_scores), dict(edge_scores), hotspot_df


# ─────────────────────────────────────────────────────────────────────────────
# 4C. Main Visualization: Canonical Spine + Deviation Overlay
# ─────────────────────────────────────────────────────────────────────────────

def visualize_deviation_overlay(G, traces, canonical_sequence, dev_results,
                                 node_scores, edge_scores, recipe_name, output_path):
    """
    The core visualization contribution.

    Layout:
      - Canonical path drawn as a bold horizontal spine through the center
      - Non-canonical nodes positioned above/below the spine
      - Deviation indicators overlaid with distinct visual encodings

    Visual encodings:
      - Node border color = intervention score (green→yellow→red heat scale)
      - Node fill = verb category (consistent with Steps 1-3)
      - Deviation glyphs: triangular markers for omissions,
        circular arrows for oscillations, crossed edges for swaps
      - Edge style: solid=canonical path, dashed=non-canonical,
        dotted+red=rare/novel transitions
    """
    print(f"\n{'═'*80}")
    print("VISUALIZING DEVIATION OVERLAY")
    print(f"{'═'*80}")

    # ── Layout: canonical path as spine ──
    # Canonical path nodes get y=0, evenly spaced on x
    pos = {}
    canon_set = set(canonical_sequence)

    # Place canonical path nodes along the x-axis
    for i, node in enumerate(canonical_sequence):
        pos[node] = (i * 3.5, 0)

    # Place non-canonical nodes above/below
    # Use median sequence position from traces
    node_seq = defaultdict(list)
    for trace in traces:
        n = max(len(trace['actions']) - 1, 1)
        for idx, action in enumerate(trace['actions']):
            node_seq[action].append(idx / n)

    non_canonical = [n for n in G.nodes() if n not in canon_set and n not in ('START', 'END')]
    max_canon_x = (len(canonical_sequence) - 1) * 3.5 if canonical_sequence else 0

    # Sort non-canonical by their median position, place alternating above/below
    non_canonical_sorted = sorted(
        non_canonical,
        key=lambda n: np.median(node_seq.get(n, [0.5]))
    )

    above = True
    for node in non_canonical_sorted:
        med_pos = np.median(node_seq.get(node, [0.5]))
        x = med_pos * max_canon_x
        y = (2.5 if above else -2.5)
        above = not above
        pos[node] = (x, y)

    # START and END
    all_x = [p[0] for p in pos.values()] if pos else [0]
    pos['START'] = (min(all_x) - 4, 0)
    pos['END'] = (max(all_x) + 4, 0)

    # ── Verb colors ──
    VERB_COLORS = {
        'take': '#3B82F6', 'put': '#8B5CF6', 'open': '#06B6D4', 'close': '#06B6D4',
        'pour': '#F97316', 'scoop': '#F97316', 'mix': '#F97316',
        'press': '#EF4444', 'crush': '#EF4444',
        'turn-on': '#10B981', 'turn-off': '#10B981',
        'wait': '#6B7280', 'check': '#6B7280', 'carry': '#3B82F6',
        'move': '#3B82F6', 'slide': '#3B82F6', 'search': '#6B7280',
        'write': '#6B7280', 'adjust': '#6B7280', 'finish': '#10B981',
    }

    def get_verb_color(node):
        if node in ('START', 'END'):
            return '#1F2937'
        return VERB_COLORS.get(node.split('(')[0].lower(), '#94A3B8')

    # ── Intervention score → border color ──
    max_score = max(node_scores.values()) if node_scores else 1.0
    if max_score == 0:
        max_score = 1.0

    def score_to_color(score):
        """Map intervention score to green→yellow→red."""
        t = min(score / max_score, 1.0)
        if t < 0.33:
            return '#22C55E'  # green
        elif t < 0.66:
            return '#EAB308'  # yellow
        else:
            return '#EF4444'  # red

    # ── Figure ──
    fig, ax = plt.subplots(figsize=(32, 14))
    ax.set_facecolor('#FAFBFC')
    fig.patch.set_facecolor('white')

    # ── Draw canonical path spine (background highlight) ──
    if len(canonical_sequence) >= 2:
        spine_xs = [pos[n][0] for n in canonical_sequence if n in pos]
        if spine_xs:
            ax.axhline(y=0, color='#E2E8F0', linewidth=8, alpha=0.5, zorder=0)
            ax.fill_between(
                [min(spine_xs) - 2, max(spine_xs) + 2],
                [-0.8, -0.8], [0.8, 0.8],
                color='#F1F5F9', alpha=0.6, zorder=0
            )
            ax.text(min(spine_xs) - 2, 0.9, 'CANONICAL PATH',
                    fontsize=8, color='#94A3B8', fontweight='bold',
                    ha='left', va='bottom')

    # ── Draw edges ──
    for u, v, data in G.edges(data=True):
        if u not in pos or v not in pos:
            continue

        is_canonical_edge = False
        for i in range(len(canonical_sequence) - 1):
            if canonical_sequence[i] == u and canonical_sequence[i + 1] == v:
                is_canonical_edge = True
                break

        # Check if START→first or last→END on canonical path
        if u == 'START' and v == canonical_sequence[0]:
            is_canonical_edge = True
        if v == 'END' and u == canonical_sequence[-1]:
            is_canonical_edge = True

        freq = data.get('frequency', 1)
        trace_frac = data.get('trace_fraction', 0)
        e_score = edge_scores.get((u, v), 0)

        if is_canonical_edge:
            # Bold, solid, dark
            lw = 2.0 + 3.0 * trace_frac
            color = '#334155'
            style = 'solid'
            alpha = 0.9
            rad = -0.02
        elif e_score > 0:
            # Deviation edge — red, dotted
            lw = 1.5
            color = '#DC2626'
            style = 'dotted'
            alpha = 0.7
            rad = 0.15
        else:
            # Non-canonical but not flagged
            lw = 0.8 + 1.5 * trace_frac
            color = '#94A3B8'
            style = 'dashed'
            alpha = max(0.2, trace_frac)
            rad = 0.1

        if u == v:
            # Self-loop
            x, y = pos[u]
            loop = mpatches.FancyArrowPatch(
                posA=(x - 0.3, y + 0.6), posB=(x + 0.3, y + 0.6),
                arrowstyle='->', connectionstyle='arc3,rad=-1.2',
                color=color, linewidth=lw, alpha=alpha, linestyle=style, zorder=2)
            ax.add_patch(loop)
        else:
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], width=lw,
                edge_color=color, alpha=alpha, arrows=True,
                arrowsize=14, arrowstyle='->', style=style,
                connectionstyle=f'arc3,rad={rad}', ax=ax,
                min_source_margin=22, min_target_margin=22)

    # ── Draw nodes ──
    max_tc = max((d.get('trace_count', 1) for _, d in G.nodes(data=True)), default=1)

    for node, data in G.nodes(data=True):
        if node not in pos:
            continue
        x, y = pos[node]
        tc = data.get('trace_count', 1)
        score = node_scores.get(node, 0)

        if node in ('START', 'END'):
            circle = plt.Circle((x, y), 0.55, color='#1F2937', zorder=4)
            ax.add_patch(circle)
            ax.text(x, y, node, fontsize=9, fontweight='bold', color='white',
                    ha='center', va='center', zorder=5)
            continue

        # Size based on trace count
        size_r = 0.35 + 0.45 * (tc / max_tc)
        fill_color = get_verb_color(node)
        border_color = score_to_color(score)
        border_width = 1.5 + 3.0 * min(score / max_score, 1.0)

        # Draw node
        circle = plt.Circle((x, y), size_r, color=fill_color,
                             ec=border_color, linewidth=border_width, zorder=4)
        ax.add_patch(circle)

        # Labels
        verb = node.split('(')[0]
        noun_part = node[len(verb):]
        ax.text(x, y, verb, fontsize=7, fontweight='bold', color='white',
                ha='center', va='center', zorder=5,
                path_effects=[pe.withStroke(linewidth=1.5, foreground=fill_color)])
        ax.text(x, y - size_r - 0.15, noun_part, fontsize=6, color='#334155',
                ha='center', va='top', zorder=5,
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])

        # Score badge (if significant)
        if score > 0.5:
            badge_color = score_to_color(score)
            ax.text(x + size_r + 0.15, y + size_r - 0.1,
                    f'{score:.1f}',
                    fontsize=6, fontweight='bold', color='white',
                    ha='center', va='center', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor=badge_color,
                              edgecolor='none', alpha=0.9))

        # ── Deviation glyphs ──
        # Omission marker (▼ triangle below node if it was frequently omitted)
        omission_count = sum(
            1 for r in dev_results
            for om in r['omissions']
            if om['action'] == node
        )
        if omission_count > 0:
            ax.plot(x, y - size_r - 0.6, marker='v', markersize=8,
                    color='#DC2626', zorder=6, markeredgecolor='white', markeredgewidth=0.5)
            ax.text(x, y - size_r - 0.85,
                    f'omitted ×{omission_count}',
                    fontsize=5, color='#DC2626', ha='center', va='top', zorder=5)

        # Oscillation marker (↻ above node if it was center of oscillations)
        osc_count = sum(
            1 for r in dev_results
            for osc in r['oscillations']
            if osc['action_a'] == node
        )
        if osc_count > 0:
            ax.text(x, y + size_r + 0.25, '↻',
                    fontsize=12, color='#F59E0B', ha='center', va='bottom',
                    fontweight='bold', zorder=6)
            ax.text(x, y + size_r + 0.6,
                    f'×{osc_count}',
                    fontsize=5.5, color='#F59E0B', ha='center', va='bottom', zorder=5)

    # ── Legend ──
    legend_elements = [
        Line2D([0], [0], color='#334155', lw=3, label='Canonical path'),
        Line2D([0], [0], color='#94A3B8', lw=1.5, linestyle='dashed', label='Non-canonical transition'),
        Line2D([0], [0], color='#DC2626', lw=1.5, linestyle='dotted', label='Flagged transition (rare/novel)'),
        Line2D([], [], marker='v', color='#DC2626', linestyle='None', markersize=8, label='Omission point'),
        Line2D([], [], marker='$↻$', color='#F59E0B', linestyle='None', markersize=12, label='Oscillation point'),
        Patch(ec='#22C55E', fc='none', lw=3, label='Border: low intervention score'),
        Patch(ec='#EAB308', fc='none', lw=3, label='Border: moderate score'),
        Patch(ec='#EF4444', fc='none', lw=3, label='Border: high score'),
    ]

    verb_legend = [
        Patch(color='#3B82F6', label='Take/carry/move'),
        Patch(color='#8B5CF6', label='Put/place'),
        Patch(color='#F97316', label='Pour/scoop/mix'),
        Patch(color='#EF4444', label='Press/crush'),
        Patch(color='#06B6D4', label='Open/close'),
        Patch(color='#10B981', label='Machine ops'),
        Patch(color='#6B7280', label='Wait/check'),
    ]

    leg1 = ax.legend(handles=legend_elements, loc='upper left', fontsize=8,
                     title='Deviation Encoding', title_fontsize=9,
                     framealpha=0.95, edgecolor='#CBD5E1')
    ax.add_artist(leg1)
    ax.legend(handles=verb_legend, loc='lower left', fontsize=8,
              title='Action Category', title_fontsize=9,
              framealpha=0.95, edgecolor='#CBD5E1')

    # ── Title ──
    n_sessions = len(traces)
    n_participants = len(set(t['participant'] for t in traces))
    total_devs = sum(
        len(r['omissions']) + len(r['oscillations']) + len(r['rare_transitions']) +
        (1 if r['ordering']['edit_distance'] > 0 else 0)
        for r in dev_results
    )

    ax.set_title(
        f'Intervention Policy Design View: {recipe_name}\n'
        f'{n_sessions} sessions · {n_participants} participant(s) · '
        f'{total_devs} deviations detected\n'
        f'Bold spine = canonical path · Border color = intervention urgency · '
        f'Glyphs = deviation type',
        fontsize=13, fontweight='bold', pad=16)
    ax.axis('off')

    all_positions = list(pos.values())
    xs = [p[0] for p in all_positions]
    ys = [p[1] for p in all_positions]
    ax.set_xlim(min(xs) - 3, max(xs) + 3)
    ax.set_ylim(min(ys) - 3, max(ys) + 3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Deviation overlay saved to {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python step4_deviation_visualization.py <recipe_name_or_id>")
        print("       python step4_deviation_visualization.py <recipe_name_or_id> --level abstracted")
        print("       python step4_deviation_visualization.py <recipe_name_or_id> --level action")
        sys.exit(1)

    try:
        recipe_query, level = parse_cli_args(sys.argv[1:])
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if not recipe_query:
        print("ERROR: Missing recipe_name_or_id.")
        sys.exit(1)

    create_output_dirs()

    # Load data from Steps 2 and 3
    canon_data, dev_data = load_step2_step3_data(recipe_query, level=level)

    G = canon_data['graph']
    traces = canon_data['traces']
    recipe_name = canon_data['meta']['name']
    level = canon_data['meta'].get('abstraction_level', level)

    dev_results = dev_data['results']
    canonical_sequence = dev_data['canonical_sequence']

    safe_name = normalize_safe_name(recipe_name)

    # Compute intervention hotspots
    node_scores, edge_scores, hotspot_df = compute_intervention_hotspots(
        G, dev_results, canonical_sequence, traces
    )

    # Save hotspot table
    hotspot_path = Path(f'../outputs/tables/intervention_hotspots_{safe_name}_{level}.csv')
    hotspot_df.to_csv(hotspot_path, index=False)
    print(f"✓ Hotspot table saved to {hotspot_path}")

    # Visualize
    fig_path = f'../outputs/figures/deviation_overlay_{safe_name}_{level}.png'
    visualize_deviation_overlay(
        G, traces, canonical_sequence, dev_results,
        node_scores, edge_scores, recipe_name, fig_path
    )

    print("\n" + "=" * 80)
    print("STEP 4 COMPLETE")
    print("=" * 80)
    print(f"\nThe deviation overlay visualization shows:")
    print(f"  • The canonical recipe path as a bold central spine")
    print(f"  • Intervention scores (node borders: green=ok, red=needs attention)")
    print(f"  • Omission markers (▼) where sessions skip critical steps")
    print(f"  • Oscillation markers (↻) where sessions show confusion")
    print(f"  • Rare/novel transitions as red dotted edges")
    print(f"\nUse this to identify WHERE a robot should intervene and WHY.")
    print(f"The hotspot CSV ranks actions by intervention priority.")


if __name__ == "__main__":
    main()