#!/usr/bin/env python3
"""
Create improved confusion matrix with:
- True Positive, False Positive, True Negative, False Negative labels
- Percentages that sum to 100% across all 4 cells
- Clear visual formatting for thesis
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11

def create_improved_confusion_matrix():
    """Create improved confusion matrix visualization."""

    # Confusion matrix values from the evaluation
    # [TN, FP]
    # [FN, TP]
    tn, fp = 12983, 611
    fn, tp = 276, 2537

    total = tn + fp + fn + tp  # 16407

    # Create figure with single improved confusion matrix
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    # Compute percentages of total (summing to 100%)
    cm = np.array([[tn, fp], [fn, tp]])
    cm_pct = cm / total * 100

    # Create heatmap
    im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)

    # Set labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Anomaly'], fontsize=12)
    ax.set_yticklabels(['Normal', 'Anomaly'], fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('Actual Label', fontsize=13, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=15)

    # Labels for each quadrant
    quadrant_labels = [
        ['True Negative\n(TN)', 'False Positive\n(FP)'],
        ['False Negative\n(FN)', 'True Positive\n(TP)']
    ]

    # Add annotations with labels, counts, and percentages
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = cm_pct[i, j]
            label = quadrant_labels[i][j]

            # Choose text color based on background
            color = 'white' if pct > 50 else 'black'

            # Create multi-line annotation
            text = f'{label}\n\n{count:,}\n({pct:.2f}%)'
            ax.text(j, i, text, ha='center', va='center',
                   color=color, fontsize=11, fontweight='bold',
                   linespacing=1.3)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Percentage of Total (%)', fontsize=11)

    # Add summary statistics below
    sensitivity = tp / (tp + fn) * 100
    specificity = tn / (tn + fp) * 100
    precision = tp / (tp + fp) * 100
    accuracy = (tp + tn) / total * 100
    f1 = 2 * (precision/100) * (sensitivity/100) / ((precision/100) + (sensitivity/100)) * 100

    stats_text = (
        f'Total Samples: {total:,}  |  '
        f'Accuracy: {accuracy:.2f}%  |  '
        f'Sensitivity: {sensitivity:.2f}%  |  '
        f'Specificity: {specificity:.2f}%  |  '
        f'F1-Score: {f1:.2f}%'
    )

    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.7))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    # Save
    output_path = Path('evaluation_plots/3_confusion_matrix_improved.png')
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', dpi=300)
    print(f"Saved: {output_path}")
    plt.close()

    # Also create a version with two matrices (raw counts and percentages)
    create_dual_confusion_matrix(cm)

    return cm


def create_dual_confusion_matrix(cm):
    """Create side-by-side confusion matrices."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    total = cm.sum()

    # Percentages of total
    cm_pct_total = cm / total * 100

    # Row-normalized (per-class percentages)
    cm_norm_rows = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Labels
    quadrant_labels = [
        ['True Negative\n(TN)', 'False Positive\n(FP)'],
        ['False Negative\n(FN)', 'True Positive\n(TP)']
    ]

    # LEFT: Raw counts with total percentages
    ax = axes[0]
    im = ax.imshow(cm_pct_total, cmap='Blues', vmin=0, vmax=100)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Anomaly'], fontsize=11)
    ax.set_yticklabels(['Normal', 'Anomaly'], fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix\n(% of Total Dataset)', fontsize=13, fontweight='bold')

    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = cm_pct_total[i, j]
            label = quadrant_labels[i][j]
            color = 'white' if pct > 50 else 'black'
            text = f'{label}\n\n{count:,}\n({pct:.2f}%)'
            ax.text(j, i, text, ha='center', va='center',
                   color=color, fontsize=10, fontweight='bold', linespacing=1.2)

    cbar1 = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar1.set_label('% of Total', fontsize=10)

    # Verify percentages sum to 100%
    print(f"Total percentage check: {cm_pct_total.sum():.2f}% (should be 100%)")

    # RIGHT: Row-normalized (sensitivity/specificity view)
    ax = axes[1]
    im = ax.imshow(cm_norm_rows, cmap='Blues', vmin=0, vmax=100)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Anomaly'], fontsize=11)
    ax.set_yticklabels(['Normal', 'Anomaly'], fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix\n(% of Actual Class)', fontsize=13, fontweight='bold')

    row_labels = [
        ['Specificity', 'False Positive\nRate'],
        ['False Negative\nRate', 'Sensitivity']
    ]

    for i in range(2):
        for j in range(2):
            pct = cm_norm_rows[i, j]
            label = row_labels[i][j]
            color = 'white' if pct > 50 else 'black'
            text = f'{label}\n\n{pct:.2f}%'
            ax.text(j, i, text, ha='center', va='center',
                   color=color, fontsize=10, fontweight='bold', linespacing=1.2)

    cbar2 = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar2.set_label('% of Actual Class', fontsize=10)

    # Summary statistics
    sensitivity = tp / (tp + fn) * 100
    specificity = tn / (tn + fp) * 100
    precision = tp / (tp + fp) * 100
    accuracy = (tp + tn) / total * 100
    f1 = 2 * (precision/100) * (sensitivity/100) / ((precision/100) + (sensitivity/100)) * 100

    stats_text = (
        f'Total: {total:,}  |  '
        f'Accuracy: {accuracy:.2f}%  |  '
        f'Precision: {precision:.2f}%  |  '
        f'Sensitivity: {sensitivity:.2f}%  |  '
        f'Specificity: {specificity:.2f}%  |  '
        f'F1: {f1:.2f}%'
    )

    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.7))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    output_path = Path('evaluation_plots/3_confusion_matrix_dual.png')
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', dpi=300)
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("Creating improved confusion matrix visualizations...")
    print("=" * 60)

    cm = create_improved_confusion_matrix()

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX SUMMARY")
    print("=" * 60)
    print(f"True Negatives:  {cm[0,0]:,} ({cm[0,0]/cm.sum()*100:.2f}%)")
    print(f"False Positives: {cm[0,1]:,} ({cm[0,1]/cm.sum()*100:.2f}%)")
    print(f"False Negatives: {cm[1,0]:,} ({cm[1,0]/cm.sum()*100:.2f}%)")
    print(f"True Positives:  {cm[1,1]:,} ({cm[1,1]/cm.sum()*100:.2f}%)")
    print(f"TOTAL:           {cm.sum():,} (100.00%)")
    print("=" * 60)
