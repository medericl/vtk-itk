import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import itk
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

from src.registration import (
    ImageType,
    load_image,
    save_image,
    build_transform,
    initialize_transform,
    register,
    resample,
    compute_quality_metrics,
)


DATA_DIR = "Data"
RESULTS_DIR = "results"
FIXED_PATH = os.path.join(DATA_DIR, "case6_gre1.nrrd")
MOVING_PATH = os.path.join(DATA_DIR, "case6_gre2.nrrd")

METHODS = ["translation", "rigid", "affine"]


def make_checkerboard(fixed, moving_registered, n_checks=8):
    """Crée une image damier alternant des blocs de fixed et de
    moving_registered, pratique pour visualiser visuellement la
    qualité de l'alignement (les structures anatomiques doivent être
    continues d'un bloc à l'autre si le recalage est bon)."""
    checker = itk.CheckerBoardImageFilter[ImageType].New()
    checker.SetInput1(fixed)
    checker.SetInput2(moving_registered)
    pattern = itk.Array[itk.UI](3)
    pattern[0] = n_checks
    pattern[1] = n_checks
    pattern[2] = n_checks
    checker.SetCheckerPattern(pattern)
    checker.Update()
    return checker.GetOutput()


def central_axial_slice(image_np):
    """Retourne la coupe axiale centrale (axe Z, premier axe numpy
    pour un volume itk.array_from_image -> ordre [z, y, x])."""
    z = image_np.shape[0] // 2
    return image_np[z, :, :]


def save_comparison_figure(fixed_np, moving_before_np, moving_after_np,
                            checker_np, method_name, out_path):
    """Génère une figure 2x2 : fixed | moving avant recalage |
    moving après recalage | checkerboard."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(central_axial_slice(fixed_np), cmap="gray")
    axes[0].set_title("Fixed (case6_gre1)")
    axes[0].axis("off")

    axes[1].imshow(central_axial_slice(moving_before_np), cmap="gray")
    axes[1].set_title("Moving avant recalage (case6_gre2)")
    axes[1].axis("off")

    axes[2].imshow(central_axial_slice(moving_after_np), cmap="gray")
    axes[2].set_title(f"Moving recalée ({method_name})")
    axes[2].axis("off")

    axes[3].imshow(central_axial_slice(checker_np), cmap="gray")
    axes[3].set_title(f"Checkerboard ({method_name})")
    axes[3].axis("off")

    fig.suptitle(f"Comparaison visuelle - méthode : {method_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_convergence_plot(histories, out_path):
    """Trace sur un même graphique les courbes de convergence des
    différents optimiseurs (une courbe par méthode), pour comparer
    vitesse et stabilité de la convergence."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, values in histories.items():
        ax.plot(values, label=method)
    ax.set_xlabel("Itération")
    ax.set_ylabel("Valeur de la métrique (Mattes Mutual Information)")
    ax.set_title("Courbes de convergence de l'optimiseur par méthode")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_metrics_bar_chart(results, out_path):
    """Diagramme en barres comparant MSE et NCC entre méthodes, plus
    le temps de calcul, pour une lecture rapide dans le rapport."""
    methods = list(results.keys())
    mse_values = [results[m]["MSE"] for m in methods]
    ncc_values = [results[m]["NCC"] for m in methods]
    time_values = [results[m]["time_sec"] for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].bar(methods, mse_values, color="indianred")
    axes[0].set_title("MSE (plus bas = mieux)")

    axes[1].bar(methods, ncc_values, color="seagreen")
    axes[1].set_title("NCC (plus proche de 1 = mieux)")
    axes[1].set_ylim(0, 1)

    axes[2].bar(methods, time_values, color="steelblue")
    axes[2].set_title("Temps de calcul (s)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Chargement des images...")
    fixed = load_image(FIXED_PATH)
    moving = load_image(MOVING_PATH)

    fixed_np = itk.array_from_image(fixed)
    moving_np = itk.array_from_image(moving)  # avant tout recalage

    results = {}
    histories = {}

    for method in METHODS:
        print(f"\n========== Méthode : {method.upper()} ==========")

        transform = build_transform(method)
        transform = initialize_transform(fixed, moving, transform)

        start = time.time()
        optimized_transform, observer, final_metric_value = register(
            fixed, moving, transform
        )
        elapsed = time.time() - start

        registered = resample(fixed, moving, optimized_transform)

        save_image(
            registered,
            os.path.join(RESULTS_DIR, f"registered_{method}.nrrd"),
        )

        quality = compute_quality_metrics(fixed, registered)
        quality["time_sec"] = elapsed
        quality["final_metric_value"] = final_metric_value
        quality["n_iterations"] = len(observer.values)
        results[method] = quality
        histories[method] = observer.values

        registered_np = itk.array_from_image(registered)
        checker_np = itk.array_from_image(
            make_checkerboard(fixed, registered)
        )

        save_comparison_figure(
            fixed_np, moving_np, registered_np, checker_np,
            method, os.path.join(RESULTS_DIR, f"comparison_{method}.png")
        )

        print(f"MSE={quality['MSE']:.2f} | NCC={quality['NCC']:.4f} | "
              f"temps={elapsed:.1f}s | itérations={quality['n_iterations']}")

    # --- Métriques "avant recalage" pour référence (baseline) ---
    baseline = compute_quality_metrics(fixed, moving)
    print(f"\n[Référence - sans recalage] MSE={baseline['MSE']:.2f} | "
          f"NCC={baseline['NCC']:.4f}")

    # --- Sauvegarde CSV récapitulatif ---
    csv_path = os.path.join(RESULTS_DIR, "comparison_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "MSE", "NCC", "time_sec",
            "final_metric_value", "n_iterations"
        ])
        writer.writerow([
            "baseline_no_registration", f"{baseline['MSE']:.4f}",
            f"{baseline['NCC']:.4f}", "0", "", ""
        ])
        for method, q in results.items():
            writer.writerow([
                method, f"{q['MSE']:.4f}", f"{q['NCC']:.4f}",
                f"{q['time_sec']:.2f}", f"{q['final_metric_value']:.6f}",
                q['n_iterations']
            ])

    save_convergence_plot(
        histories, os.path.join(RESULTS_DIR, "convergence_curves.png")
    )
    save_metrics_bar_chart(
        results, os.path.join(RESULTS_DIR, "metrics_comparison.png")
    )

    # --- Détermination automatique de la "meilleure" méthode (NCC max) ---
    best_method = max(results, key=lambda m: results[m]["NCC"])

    print("\n========== RÉCAPITULATIF ==========")
    print(f"{'Méthode':<15}{'MSE':>12}{'NCC':>10}{'Temps(s)':>10}")
    print(f"{'baseline':<15}{baseline['MSE']:>12.2f}{baseline['NCC']:>10.4f}"
          f"{'-':>10}")
    for method, q in results.items():
        print(f"{method:<15}{q['MSE']:>12.2f}{q['NCC']:>10.4f}"
              f"{q['time_sec']:>10.1f}")
    print(f"\n=> Meilleure méthode (NCC le plus élevé) : {best_method}")
    print(f"\nRésultats sauvegardés dans : {RESULTS_DIR}/")
    print(" - registered_<methode>.nrrd   : volumes recalés")
    print(" - comparison_<methode>.png    : comparaison visuelle")
    print(" - convergence_curves.png      : courbes de convergence")
    print(" - metrics_comparison.png      : barres MSE/NCC/temps")
    print(" - comparison_metrics.csv      : tableau récapitulatif")


if __name__ == "__main__":
    main()
