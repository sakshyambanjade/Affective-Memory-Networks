import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(10, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

# Define boxes: (x, y, label, color)
boxes = [
    (5, 13, "User Input", "#3498db"),
    (5, 11.5, "Emotional Appraisal\n(VAD + Lazarus)", "#e74c3c"),
    (5, 10, "Working Memory\n(capacity=5)", "#2ecc71"),
    (5, 8.5, "Episodic Memory\n(long-term)", "#9b59b6"),
    (5, 7, "Retrieval Engine\n(5 factors)", "#f39c12"),
    (5, 5.5, "LLM Context", "#1abc9c"),
    (5, 4, "Agent Response", "#3498db"),
]

# Draw boxes
for x, y, label, color in boxes:
    box = FancyBboxPatch((x-1.5, y-0.4), 3, 0.8,
                        boxstyle="round,pad=0.1",
                        edgecolor='black', facecolor=color,
                        linewidth=2, alpha=0.7)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

# Draw arrows
arrow_pairs = [(13, 11.5), (11.5, 10), (10, 8.5), (8.5, 7), (7, 5.5), (5.5, 4)]
for y1, y2 in arrow_pairs:
    arrow = FancyArrowPatch((5, y1-0.5), (5, y2+0.5),
                           arrowstyle='->', mutation_scale=30,
                           linewidth=3, color='black')
    ax.add_patch(arrow)

# Add side annotations
ax.text(8, 11.5, "V, A, D scores", fontsize=9, style='italic')
ax.text(8, 10, "Recent 5 turns", fontsize=9, style='italic')
ax.text(8, 8.5, "All past turns", fontsize=9, style='italic')
ax.text(8, 7, "Multi-factor scoring", fontsize=9, style='italic')

plt.title("AMN Architecture", fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
from pathlib import Path
figures_dir = Path('results/figures')
figures_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(figures_dir / 'figure3_architecture.png', dpi=300, bbox_inches='tight')
print("Saved: results/figure3_architecture.png")
