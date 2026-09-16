import itk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def fig_slices_grid(image_t1, image_t2, mask_t1, mask_t2, output_path, y_center=80, n_slices=5, step=6):
    RED  = (1.0, 0.15, 0.15)
    BLUE = (0.15, 0.45, 1.0)

    arr_t1 = itk.array_from_image(image_t1).astype(float)
    arr_t2 = itk.array_from_image(image_t2).astype(float)
    arr_m1 = itk.array_from_image(mask_t1).astype(float)
    arr_m2 = itk.array_from_image(mask_t2).astype(float)

    ys = [y_center + (i - n_slices // 2) * step for i in range(n_slices)]

    lo1, hi1 = np.percentile(arr_t1[arr_t1 > 0], [1, 99])
    lo2, hi2 = np.percentile(arr_t2[arr_t2 > 0], [1, 99])

    fig, axes = plt.subplots(2, n_slices, figsize=(3 * n_slices, 6))
    fig.suptitle("Évolution tumorale — coupes axiales autour de la tumeur",
                 fontsize=13, fontweight='bold')

    for col, y in enumerate(ys):
        sl1 = arr_t1[:, y, :]; m1 = arr_m1[:, y, :]
        sl2 = arr_t2[:, y, :]; m2 = arr_m2[:, y, :]

        def overlay(ax, sl, mask, color):
            rgba = np.zeros((*mask.shape, 4), dtype=float)
            rgba[mask > 0, :3] = color
            rgba[mask > 0,  3] = 0.45
            ax.imshow(rgba, origin='lower', aspect='equal')

        ax = axes[0, col]
        ax.imshow(sl1, cmap='gray', vmin=lo1, vmax=hi1, origin='lower', aspect='equal')
        overlay(ax, sl1, m1, RED)
        ax.set_title(f"y={y}", fontsize=9)
        ax.axis('off')
        if col == 0:
            ax.set_ylabel("T1", fontsize=10, rotation=0, labelpad=30, va='center')

        ax = axes[1, col]
        ax.imshow(sl2, cmap='gray', vmin=lo2, vmax=hi2, origin='lower', aspect='equal')
        overlay(ax, sl2, m2, BLUE)
        ax.axis('off')
        if col == 0:
            ax.set_ylabel("T2", fontsize=10, rotation=0, labelpad=30, va='center')

    p1 = mpatches.Patch(color=RED,  alpha=0.7, label="T1")
    p2 = mpatches.Patch(color=BLUE, alpha=0.7, label="T2")
    fig.legend(handles=[p1, p2], loc='lower center', ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   -> {output_path}")
