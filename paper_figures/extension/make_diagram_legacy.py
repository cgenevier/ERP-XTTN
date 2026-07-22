"""
ERP-XTTN architecture diagram for the multi-ERP extension (auto mode).
Adapted from the Graz figure (constrained mode). Edits reflect
the data-driven configuration shared across all 9 datasets in the paper:
auto peak detection (no polarity priors), prominence threshold 0.02, dynamic
K ≤ 4, dataset-specific detection channel, and a generic class output.

Outputs: erp_xttn_architecture.pdf (vector) and erp_xttn_architecture.png (raster preview)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

# --- Font sizes (bumped from original) ---
FS_TITLE      = 18   # "Error / Correct", "Prototype Construction"
FS_BOX_MAIN   = 14   # Main box labels
FS_BOX_SUB    = 11   # Sub-text within boxes (e.g., "Linear -> logit")
FS_TAG        = 13   # Q, K, H labels on arrows
FS_ITALIC     = 12   # "interpretable", "shared weights", prototype notes

# --- Colors (match original) ---
C_BLUE_FILL   = "#DDEBF7"
C_BLUE_EDGE   = "#2E5C8A"
C_GREEN_FILL  = "#E8F0E4"
C_GREEN_EDGE  = "#6B8E5A"
C_GRAY_FILL   = "#F2F2F2"
C_GRAY_EDGE   = "#999999"
C_WHITE       = "#FFFFFF"
C_ORANGE      = "#C75B3D"
C_ORANGE_FILL = "#FBEEE8"
C_BLACK       = "#000000"

# --- Figure: keep proportions similar to original (taller than wide) ---
fig, ax = plt.subplots(figsize=(11, 17))
ax.set_xlim(0, 100)
ax.set_ylim(0, 170)
ax.set_aspect('equal')
ax.axis('off')

def box(x, y, w, h, main_text, sub_text=None, fill=C_WHITE, edge=C_GRAY_EDGE,
        lw=1.5, main_fs=FS_BOX_MAIN, sub_fs=FS_BOX_SUB, main_weight='normal',
        main_color='black', sub_color='#555555', rounding=0.8):
    """Draw a rounded box with optional sub-text."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        linewidth=lw, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(patch)
    if sub_text:
        # main text slightly above center, sub slightly below
        ax.text(x + w/2, y + h*0.62, main_text, ha='center', va='center',
                fontsize=main_fs, fontweight=main_weight, color=main_color)
        ax.text(x + w/2, y + h*0.28, sub_text, ha='center', va='center',
                fontsize=sub_fs, color=sub_color)
    else:
        ax.text(x + w/2, y + h/2, main_text, ha='center', va='center',
                fontsize=main_fs, fontweight=main_weight, color=main_color)

def arrow(x1, y1, x2, y2, color='black', lw=1.5, style='-|>', mutation=18):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=style, mutation_scale=mutation,
                        color=color, lw=lw, shrinkA=0, shrinkB=0)
    ax.add_patch(a)

# =====================================================================
# LEFT COLUMN (main pipeline): x roughly 8..52, width ~44
# =====================================================================
LEFT_X = 8
LEFT_W = 44
LEFT_CX = LEFT_X + LEFT_W/2  # 30

# --- EEG Epoch (bottom) ---
y = 4
box(LEFT_X, y, LEFT_W, 6, "EEG Epoch  (C × T)", fill=C_GRAY_FILL, edge=C_GRAY_EDGE)
eeg_top = y + 6

# arrow up to Patch Embedding
arrow(LEFT_CX, eeg_top, LEFT_CX, eeg_top + 4)

# --- Patch Embedding ---
y = 14
box(LEFT_X, y, LEFT_W, 8, "Patch Embedding",
    "Linear(C×8, 64),  non-overlapping,  pw = 8",
    edge=C_GRAY_EDGE)
pe_top = y + 8

# arrow up to Positional Embedding
arrow(LEFT_CX, pe_top, LEFT_CX, pe_top + 3)

# --- Positional Embedding ---
y = 25
box(LEFT_X, y, LEFT_W, 8, "Positional Embedding",
    "learned,  shared with prototype path",
    edge=C_GRAY_EDGE)
pos_top = y + 8

# small circle (tokens) above positional emb
circ_y = pos_top + 3.5
ax.add_patch(Circle((LEFT_CX, circ_y), 0.8, fill=False, edgecolor='black', lw=1.3))
arrow(LEFT_CX, pos_top, LEFT_CX, circ_y - 0.8, mutation=14)

# --- Transformer block (gray surrounding box) ---
tb_x, tb_y, tb_w, tb_h = LEFT_X - 1, 42, LEFT_W + 2, 32
tb_patch = FancyBboxPatch((tb_x, tb_y), tb_w, tb_h,
                          boxstyle="round,pad=0,rounding_size=0.8",
                          linewidth=1.2, edgecolor='#BBBBBB', facecolor='#FAFAFA')
ax.add_patch(tb_patch)

# arrow from circle up into LayerNorm
arrow(LEFT_CX, circ_y + 0.8, LEFT_CX, 44.5)

# --- LayerNorm ---
y = 44.5
box(LEFT_X, y, LEFT_W, 5, "LayerNorm", fill=C_WHITE, edge=C_GRAY_EDGE)
ln_top = y + 5
arrow(LEFT_CX, ln_top, LEFT_CX, ln_top + 2)

# --- Multi-Head Self-Attention ---
y = 52
box(LEFT_X, y, LEFT_W, 8, "Multi-Head Self-Attention",
    "H = 4,  d_head = 16",
    fill=C_BLUE_FILL, edge=C_BLUE_EDGE, lw=2)
mhsa_top = y + 8
arrow(LEFT_CX, mhsa_top, LEFT_CX, mhsa_top + 2)

# --- Output Projection ---
y = 62
box(LEFT_X, y, LEFT_W, 6, "Output Projection", fill=C_WHITE, edge=C_GRAY_EDGE)
op_top = y + 6

# arrow from transformer block top up to plus symbol
arrow(LEFT_CX, op_top, LEFT_CX, op_top + 4, mutation=16)

# --- Residual plus symbol ---
plus_y = 76
ax.add_patch(Circle((LEFT_CX, plus_y), 1.4, fill=False, edgecolor='black', lw=1.4))
ax.text(LEFT_CX, plus_y, "+", ha='center', va='center', fontsize=16, fontweight='bold')

# residual skip connection: from circle (tokens, at circ_y) -> LEFT side -> up -> into plus
# left-side path (mirrors original right-side routing but on the left to
# keep the right side of the main column clear for shared-weights)
skip_x = LEFT_X - 2
# horizontal from tokens circle to the left
ax.plot([LEFT_CX - 0.8, skip_x], [circ_y, circ_y], color='#666666', lw=1.2)
# vertical up
ax.plot([skip_x, skip_x], [circ_y, plus_y], color='#666666', lw=1.2)
# horizontal right into plus with arrow
arrow(skip_x, plus_y, LEFT_CX - 1.4, plus_y, color='#666666', lw=1.2, mutation=14)

# arrow from plus up to Cross-Attention
arrow(LEFT_CX, plus_y + 1.4, LEFT_CX, plus_y + 5, mutation=16)

# Q label on that arrow (left side, to leave the right side clear for
# the shared-weights dotted connector)
ax.text(LEFT_CX - 3.5, plus_y + 3, "Q", ha='right', va='center',
        fontsize=FS_TAG, fontweight='bold')

# --- Cross-Attention (inside green "interpretable" box) ---
green_x, green_y, green_w, green_h = LEFT_X - 2, 82, LEFT_W + 4, 16
green_patch = FancyBboxPatch((green_x, green_y), green_w, green_h,
                             boxstyle="round,pad=0,rounding_size=0.8",
                             linewidth=1.5, edgecolor=C_GREEN_EDGE,
                             facecolor=C_GREEN_FILL)
ax.add_patch(green_patch)
ax.text(green_x + green_w/2, green_y + green_h - 2, "interpretable",
        ha='center', va='center', fontsize=FS_ITALIC, style='italic',
        color=C_GREEN_EDGE)

# Cross-Attention inner box
box(LEFT_X, 85, LEFT_W, 9, "Cross-Attention (QK-only)",
    "H = 4,  no V projection",
    fill=C_BLUE_FILL, edge=C_BLUE_EDGE, lw=2)

# arrow from cross-attn up to Head Average
arrow(LEFT_CX, 98, LEFT_CX, 102, mutation=16)

# --- Head Average + Flatten ---
box(LEFT_X, 102, LEFT_W, 6, "Head Average + Flatten",
    fill=C_GRAY_FILL, edge=C_GRAY_EDGE)
arrow(LEFT_CX, 108, LEFT_CX, 112, mutation=16)

# --- Classifier ---
box(LEFT_X, 112, LEFT_W, 9, "Classifier", "Linear → logit",
    fill=C_WHITE, edge=C_BLACK, lw=2)

# arrow up to output label
arrow(LEFT_CX, 121, LEFT_CX, 126, mutation=18)

# --- Output label ---
ax.text(LEFT_CX, 129, "Class probability", ha='center', va='center',
        fontsize=FS_TITLE + 2, fontweight='bold')

# =====================================================================
# RIGHT COLUMN: Prototype Construction
# =====================================================================
RIGHT_X = 58
RIGHT_W = 38
RIGHT_CX = RIGHT_X + RIGHT_W/2  # 77

# Outer orange container
proto_y = 8
proto_h = 115
proto_outer = FancyBboxPatch((RIGHT_X - 1, proto_y), RIGHT_W + 2, proto_h,
                             boxstyle="round,pad=0,rounding_size=0.8",
                             linewidth=1.8, edgecolor=C_ORANGE,
                             facecolor=C_ORANGE_FILL, alpha=0.45)
ax.add_patch(proto_outer)

# Title
ax.text(RIGHT_CX, proto_y + proto_h - 4, "Prototype Construction",
        ha='center', va='center', fontsize=FS_TITLE, color=C_ORANGE,
        fontweight='normal')
ax.text(RIGHT_CX, proto_y + proto_h - 9, "frozen; recomputed twice per LOSO fold",
        ha='center', va='center', fontsize=FS_ITALIC, style='italic',
        color=C_ORANGE)
ax.text(RIGHT_CX, proto_y + proto_h - 12.5,
        "(Phase 1 split, Phase 2 full non-test pool)",
        ha='center', va='center', fontsize=FS_ITALIC, style='italic',
        color=C_ORANGE)

# --- Bottom: Train-set Grand-Avg Diff Wave ---
y = 12
box(RIGHT_X, y, RIGHT_W, 8, "Train-set Grand-Avg Diff Wave",
    "pos − neg class;  detection channel for peaks",
    fill=C_WHITE, edge=C_ORANGE, lw=1.5)
arrow(RIGHT_CX, y + 8, RIGHT_CX, y + 11, color=C_ORANGE, mutation=16)

# --- Peak Detection (two sub-lines) ---
y = 23
box_patch = FancyBboxPatch((RIGHT_X, y), RIGHT_W, 11,
                           boxstyle="round,pad=0,rounding_size=0.8",
                           linewidth=1.5, edgecolor=C_ORANGE, facecolor=C_WHITE)
ax.add_patch(box_patch)
ax.text(RIGHT_CX, y + 8.5, "Peak Detection", ha='center', va='center',
        fontsize=FS_BOX_MAIN, color='black')
ax.text(RIGHT_CX, y + 5, "auto, top-K by prominence,  ≥ 50 ms latency",
        ha='center', va='center', fontsize=FS_BOX_SUB, color='#555555')
ax.text(RIGHT_CX, y + 2, "prom. ≥ 0.02,  zero-cross. (40–200 ms)",
        ha='center', va='center', fontsize=FS_BOX_SUB, color='#555555')
arrow(RIGHT_CX, y + 11, RIGHT_CX, y + 14, color=C_ORANGE, mutation=16)

# --- K = 4 Windowed Prototypes ---
y = 38
box(RIGHT_X, y, RIGHT_W, 7, "K ≤ 4  Windowed Prototypes",
    fill=C_WHITE, edge=C_ORANGE, lw=1.5)
arrow(RIGHT_CX, y + 7, RIGHT_CX, y + 10, color=C_ORANGE, mutation=16)

# --- Shared Patch Embedding ---
y = 48
box(RIGHT_X, y, RIGHT_W, 7, "Shared Patch Embedding",
    fill=C_WHITE, edge=C_ORANGE, lw=1.5)
arrow(RIGHT_CX, y + 7, RIGHT_CX, y + 10, color=C_ORANGE, mutation=16)

# --- Mean Pool + Center Pos. Embed. ---
y = 58
box(RIGHT_X, y, RIGHT_W, 7, "Mean Pool + Center Pos. Embed.",
    fill=C_WHITE, edge=C_ORANGE, lw=1.5)

# --- "shared weights" brackets pairing {Patch Emb, Pos Emb} on the left
#     with {Shared Patch Emb, Mean Pool + Center Pos Emb} on the right ---
#
# Left-side boxes (main pipeline):
#   Patch Embedding:       y = 14, h = 8   -> spans 14..22
#   Positional Embedding:  y = 25, h = 8   -> spans 25..33
# Right-side boxes (prototype path):
#   Shared Patch Embedding:        y = 48, h = 7   -> spans 48..55
#   Mean Pool + Center Pos Embed:  y = 58, h = 7   -> spans 58..65
#
# A bracket is drawn on the right edge of the left pair (spanning both boxes)
# and on the left edge of the right pair (spanning both boxes), connected by
# a dotted line labeled "shared weights".

BR_COLOR = '#888888'
BR_LW = 1.2
BR_TICK = 1.2  # how far the bracket tips stick inward

# Left bracket: right edge of left pair, covering 14..33
LB_X = LEFT_X + LEFT_W + 1.5   # just right of the left column
LB_TOP = 33                    # top of Positional Embedding
LB_BOT = 14                    # bottom of Patch Embedding
LB_MID = (LB_TOP + LB_BOT) / 2
# vertical spine
ax.plot([LB_X, LB_X], [LB_BOT, LB_TOP], color=BR_COLOR, lw=BR_LW)
# top tick pointing left into the Pos Emb box
ax.plot([LB_X, LB_X - BR_TICK], [LB_TOP, LB_TOP], color=BR_COLOR, lw=BR_LW)
# bottom tick pointing left into the Patch Emb box
ax.plot([LB_X, LB_X - BR_TICK], [LB_BOT, LB_BOT], color=BR_COLOR, lw=BR_LW)
# middle tick pointing right (toward the connector line)
ax.plot([LB_X, LB_X + BR_TICK], [LB_MID, LB_MID], color=BR_COLOR, lw=BR_LW)

# Right bracket: left edge of right pair, covering 48..65
RB_X = RIGHT_X - 1.5
RB_TOP = 65                    # top of Mean Pool box
RB_BOT = 48                    # bottom of Shared Patch Embedding
RB_MID = (RB_TOP + RB_BOT) / 2
ax.plot([RB_X, RB_X], [RB_BOT, RB_TOP], color=BR_COLOR, lw=BR_LW)
# top tick pointing right into the Mean Pool box
ax.plot([RB_X, RB_X + BR_TICK], [RB_TOP, RB_TOP], color=BR_COLOR, lw=BR_LW)
# bottom tick pointing right into the Shared Patch Emb box
ax.plot([RB_X, RB_X + BR_TICK], [RB_BOT, RB_BOT], color=BR_COLOR, lw=BR_LW)
# middle tick pointing left (toward the connector line)
ax.plot([RB_X, RB_X - BR_TICK], [RB_MID, RB_MID], color=BR_COLOR, lw=BR_LW)

# Dotted connector between the two bracket midpoints
ax.plot([LB_X + BR_TICK, RB_X - BR_TICK], [LB_MID, RB_MID],
        linestyle=':', color=BR_COLOR, lw=BR_LW)

import numpy as np
# Label: rotated to ride along the dotted connector.
# Connector runs from (LB_X+BR_TICK, LB_MID) to (RB_X-BR_TICK, RB_MID).
x1, y1 = LB_X + BR_TICK, LB_MID
x2, y2 = RB_X - BR_TICK, RB_MID
# Midpoint of the connector
mx, my = (x1 + x2) / 2, (y1 + y2) / 2
# Angle in data coords; aspect is equal so this matches display angle.
angle_deg = np.degrees(np.arctan2(y2 - y1, x2 - x1))
# Small perpendicular offset so the text sits just above the line rather
# than directly on it (perpendicular unit vector to the connector).
dx, dy = x2 - x1, y2 - y1
length = np.hypot(dx, dy)
perp_x, perp_y = -dy / length, dx / length  # rotate 90° CCW
offset = 1.2
tx = mx + perp_x * offset
ty = my + perp_y * offset
ax.text(tx, ty, "shared weights",
        ha='center', va='center', fontsize=FS_ITALIC, style='italic',
        color='#555555', rotation=angle_deg, rotation_mode='anchor')

# --- Arrow from prototype stack up-and-left into Cross-Attention (K) ---
# originates near top of rightmost column at Mean Pool top (y = 65)
# goes up along right side then left into cross-attn
start_x = RIGHT_CX
start_y = 65
# up along the right column
ax.plot([start_x, start_x], [start_y, 89.5], color=C_ORANGE, lw=1.8)
# left into cross-attn box with arrowhead
arrow(start_x, 89.5, LEFT_X + LEFT_W + 0.2, 89.5, color=C_ORANGE, lw=1.8,
      mutation=20)

# K label on the horizontal arrow (bold red)
ax.text((LEFT_X + LEFT_W + start_x)/2 + 2, 92, "K",
        ha='center', va='center', fontsize=FS_TAG + 2,
        fontweight='bold', color=C_ORANGE)

# =====================================================================
# Save
# =====================================================================
import os
HERE = os.path.dirname(os.path.abspath(__file__))
plt.tight_layout()
plt.savefig(os.path.join(HERE, 'erp_xttn_architecture.pdf'),
            bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(HERE, 'erp_xttn_architecture.png'),
            bbox_inches='tight', dpi=200)
print("Saved PDF and PNG to", HERE)
