#!/usr/bin/env python3
"""Create an improved pipeline diagram for the ECG anomaly detection approach."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_pipeline_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 14)
    ax.axis('off')
    ax.set_aspect('equal')

    # Colors
    colors = {
        'data': '#E3F2FD',       # Light blue
        'preprocess': '#E8F5E9', # Light green
        'model': '#FFF3E0',      # Light orange
        'train': '#FCE4EC',      # Light pink
        'test': '#E8EAF6',       # Light indigo
        'eval': '#F3E5F5',       # Light purple
        'output': '#E0F7FA',     # Light cyan
        'arrow': '#455A64',      # Dark gray
        'border': '#37474F',     # Darker gray
    }

    def draw_box(x, y, width, height, text, color, fontsize=10, bold=False):
        """Draw a rounded rectangle with text."""
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            facecolor=color,
            edgecolor=colors['border'],
            linewidth=2
        )
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(x + width/2, y + height/2, text,
                ha='center', va='center', fontsize=fontsize,
                fontweight=weight, wrap=True)

    def draw_arrow(x1, y1, x2, y2, curved=False, rad=0.0):
        """Draw an arrow between two points."""
        if curved:
            style = f"arc3,rad={rad}"
        else:
            style = "arc3,rad=0"
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=style,
            arrowstyle='->,head_width=0.3,head_length=0.2',
            color=colors['arrow'],
            linewidth=2,
            mutation_scale=15
        )
        ax.add_patch(arrow)

    # Title
    ax.text(9, 13.5, 'ECG Anomaly Detection Pipeline',
            ha='center', va='center', fontsize=20, fontweight='bold')
    ax.text(9, 13.0, 'Convolutional Autoencoder Approach',
            ha='center', va='center', fontsize=13, fontstyle='italic')

    # ===== ROW 1: Data Collection =====
    y1 = 11.0
    draw_box(0.5, y1, 2.5, 1.2, 'MIT-BIH\nArrhythmia\nDatabase', colors['data'], bold=True)
    draw_box(4, y1, 2.2, 1.2, '48 Records\n360 Hz\n~110K beats', colors['data'])
    draw_box(7.2, y1, 2.2, 1.2, 'Beat\nAnnotations\n(Cardiologist)', colors['data'])

    # Section label
    ax.text(0.3, y1 + 1.5, '1. DATA COLLECTION', fontsize=12, fontweight='bold', color='#1565C0')

    # Arrows Row 1
    draw_arrow(3.0, y1 + 0.6, 4.0, y1 + 0.6)
    draw_arrow(6.2, y1 + 0.6, 7.2, y1 + 0.6)

    # ===== ROW 2: Preprocessing =====
    y2 = 8.7
    draw_box(0.5, y2, 2.0, 1.2, '2-Second\nWindowing', colors['preprocess'])
    draw_box(3.0, y2, 2.0, 1.2, 'Bandpass\nFilter\n(0.5-40 Hz)', colors['preprocess'])
    draw_box(5.5, y2, 2.0, 1.2, 'Z-Score\nNormalization', colors['preprocess'])
    draw_box(8.0, y2, 2.5, 1.2, 'Label\nAssignment\n(Normal/Anomaly)', colors['preprocess'])

    ax.text(0.3, y2 + 1.5, '2. PREPROCESSING', fontsize=12, fontweight='bold', color='#2E7D32')

    # Arrow from Row 1 to Row 2
    draw_arrow(5.1, y1, 1.5, y2 + 1.2)
    # Arrows within Row 2
    draw_arrow(2.5, y2 + 0.6, 3.0, y2 + 0.6)
    draw_arrow(5.0, y2 + 0.6, 5.5, y2 + 0.6)
    draw_arrow(7.5, y2 + 0.6, 8.0, y2 + 0.6)

    # ===== ROW 3: Data Split =====
    y3 = 6.4
    draw_box(0.5, y3, 2.8, 1.2, 'Train Set (70%)\nNormal Beats Only', colors['train'])
    draw_box(4.0, y3, 2.8, 1.2, 'Validation Set (15%)\nNormal Beats Only', colors['train'])
    draw_box(7.5, y3, 2.8, 1.2, 'Test Set (15%)\nAll Beats (Mixed)', colors['test'])

    ax.text(0.3, y3 + 1.5, '3. DATA SPLIT', fontsize=12, fontweight='bold', color='#C62828')

    # Arrow from Preprocessing to Data Split (center arrow going down)
    draw_arrow(9.25, y2, 5.4, y3 + 1.2)

    # ===== ROW 4: Model Architecture & Training =====
    y4 = 3.8

    # Encoder box
    draw_box(0.5, y4, 2.5, 1.5, 'ENCODER\nConv1D Layers\n720 → 32', colors['model'], bold=True)

    # Latent space
    draw_box(3.5, y4 + 0.15, 1.5, 1.2, 'Latent\nSpace\n(32-dim)', colors['model'])

    # Decoder box
    draw_box(5.5, y4, 2.5, 1.5, 'DECODER\nConvT1D Layers\n32 → 720', colors['model'], bold=True)

    # Training components
    draw_box(9.0, y4, 2.8, 1.5, 'Training Loop\n• MSE Loss\n• AdamW Optimizer\n• Early Stopping', colors['train'])

    # Validation arrow and box
    draw_box(12.5, y4, 2.5, 1.5, 'Validation\n• Monitor Loss\n• Save Best Model', colors['train'])

    ax.text(0.3, y4 + 1.8, '4. MODEL TRAINING', fontsize=12, fontweight='bold', color='#E65100')

    # Arrows for training flow
    draw_arrow(1.9, y3, 1.75, y4 + 1.5)  # Train set to Encoder
    draw_arrow(3.0, y4 + 0.75, 3.5, y4 + 0.75)  # Encoder to Latent
    draw_arrow(5.0, y4 + 0.75, 5.5, y4 + 0.75)  # Latent to Decoder
    draw_arrow(8.0, y4 + 0.75, 9.0, y4 + 0.75)  # Decoder to Training
    draw_arrow(11.8, y4 + 0.75, 12.5, y4 + 0.75)  # Training to Validation
    draw_arrow(5.4, y3, 13.75, y4 + 1.5)  # Validation set to Validation box

    # ===== ROW 5: Testing =====
    y5 = 1.8
    draw_box(0.5, y5, 2.5, 1.2, 'Load Best\nTrained Model', colors['test'])
    draw_box(3.5, y5, 2.5, 1.2, 'Forward Pass\non Test Set', colors['test'])
    draw_box(6.5, y5, 2.5, 1.2, 'Compute\nReconstruction\nError (MSE)', colors['test'])

    ax.text(0.3, y5 + 1.5, '5. TESTING', fontsize=12, fontweight='bold', color='#303F9F')

    # Arrow from Test set to Testing phase
    draw_arrow(8.9, y3, 8.9, y5 + 2.5, curved=True, rad=-0.3)
    ax.annotate('', xy=(1.75, y5 + 1.2), xytext=(8.9, y5 + 2.5),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))

    # Arrows in Testing row
    draw_arrow(3.0, y5 + 0.6, 3.5, y5 + 0.6)
    draw_arrow(6.0, y5 + 0.6, 6.5, y5 + 0.6)

    # ===== ROW 6: Evaluation =====
    y6 = -0.2
    draw_box(0.5, y6, 2.3, 1.2, 'Threshold\nSelection\n(Max F1)', colors['eval'])
    draw_box(3.3, y6, 2.3, 1.2, 'Classification\nNormal vs\nAnomaly', colors['eval'])
    draw_box(6.1, y6, 2.8, 1.2, 'Metrics\nROC-AUC, PR-AUC\nF1, Precision, Recall', colors['eval'])

    # Target box (highlighted)
    target_box = FancyBboxPatch(
        (9.5, y6 - 0.1), 3.2, 1.4,
        boxstyle="round,pad=0.05,rounding_size=0.2",
        facecolor='#C8E6C9',  # Lighter green
        edgecolor='#2E7D32',  # Dark green border
        linewidth=3
    )
    ax.add_patch(target_box)
    ax.text(11.1, y6 + 0.6, 'TARGET\nROC-AUC ≥ 99%', ha='center', va='center',
            fontsize=12, fontweight='bold', color='#1B5E20')

    ax.text(0.3, y6 + 1.5, '6. EVALUATION', fontsize=12, fontweight='bold', color='#6A1B9A')

    # Arrows in Evaluation row
    draw_arrow(7.75, y5, 1.65, y6 + 1.2)  # From MSE to Threshold
    draw_arrow(2.8, y6 + 0.6, 3.3, y6 + 0.6)
    draw_arrow(5.6, y6 + 0.6, 6.1, y6 + 0.6)
    draw_arrow(8.9, y6 + 0.6, 9.5, y6 + 0.6)

    # ===== Legend =====
    legend_y = 5.5
    legend_x = 15.2
    ax.text(legend_x, legend_y + 2.2, 'Legend:', fontsize=11, fontweight='bold')

    legend_items = [
        ('Data Collection', colors['data']),
        ('Preprocessing', colors['preprocess']),
        ('Training', colors['train']),
        ('Model', colors['model']),
        ('Testing', colors['test']),
        ('Evaluation', colors['eval']),
        ('Target', '#C8E6C9'),
    ]

    for i, (label, color) in enumerate(legend_items):
        y_pos = legend_y + 1.7 - i * 0.45
        box = FancyBboxPatch(
            (legend_x, y_pos - 0.15), 0.5, 0.35,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor=colors['border'],
            linewidth=1
        )
        ax.add_patch(box)
        ax.text(legend_x + 0.6, y_pos, label, fontsize=10, va='center')

    # Add note about 99% target
    ax.text(15.2, 2.5, 'Why 99% ROC-AUC?', fontsize=10, fontweight='bold', color='#1B5E20')
    note_text = ('Medical diagnostic\nstandards require\nhigh sensitivity to\nminimize missed\narrhythmias')
    ax.text(15.2, 1.5, note_text, fontsize=9, va='top', color='#37474F')

    plt.tight_layout()
    plt.savefig('pipeline_diagram.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved pipeline_diagram.png")
    plt.close()

if __name__ == "__main__":
    create_pipeline_diagram()
