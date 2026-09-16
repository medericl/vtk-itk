import os

from src.registration import (
    load_image,
    save_image,
    build_transform,
    initialize_transform,
    register,
    resample,
    compute_quality_metrics,
)
from src.segmentation import segment_tumor, save_mask, count_voxels, voxel_volume_mm3
from src.analysis import compare_tumors, print_report
from src.visualization import display_comparison
from src.figures import fig_slices_grid


DATA_DIR = "Data"
RESULTS_DIR = "results"
FIXED_PATH = os.path.join(DATA_DIR, "case6_gre1.nrrd")
MOVING_PATH = os.path.join(DATA_DIR, "case6_gre2.nrrd")
REGISTRATION_METHOD = "rigid"
SEED_FIXED  = (90, 80, 47)
SEED_MOVING = (90, 80, 47)


def run_registration(fixed, moving):
    print(f"\n[1/4] Recalage ({REGISTRATION_METHOD})...")
    transform = build_transform(REGISTRATION_METHOD)
    transform = initialize_transform(fixed, moving, transform)
    optimized_transform, _, _ = register(fixed, moving, transform)
    registered = resample(fixed, moving, optimized_transform)

    quality = compute_quality_metrics(fixed, registered)
    print(f"   -> MSE={quality['MSE']:.2f}, NCC={quality['NCC']:.4f}")

    save_image(registered, os.path.join(RESULTS_DIR, "registered.nrrd"))
    return registered


def run_segmentation(fixed, registered_moving):
    print("\n[2/4] Segmentation des tumeurs (méthode : Otsu 2 étapes + composante connexe)...")
    mask_fixed  = segment_tumor(fixed,             seed=SEED_FIXED)
    mask_moving = segment_tumor(registered_moving, seed=SEED_MOVING)

    n1 = count_voxels(mask_fixed)
    n2 = count_voxels(mask_moving)
    vox_vol = voxel_volume_mm3(mask_fixed)
    print(f"   -> T1 : {n1} voxels ({n1 * vox_vol:.0f} mm³)")
    print(f"   -> T2 : {n2} voxels ({n2 * vox_vol:.0f} mm³)")

    save_mask(mask_fixed, os.path.join(RESULTS_DIR, "mask_t1.nrrd"))
    save_mask(mask_moving, os.path.join(RESULTS_DIR, "mask_t2.nrrd"))
    return mask_fixed, mask_moving


def run_analysis(fixed, registered_moving, mask_fixed, mask_moving):
    print("\n[3/4] Analyse des changements...")
    metrics = compare_tumors(fixed, registered_moving, mask_fixed, mask_moving)
    print_report(metrics)
    return metrics


def run_visualisation(fixed, registered_moving, mask_fixed, mask_moving):
    print("\n[4/4] Visualisation...")

    print("   [4a] Figures matplotlib...")
    fig_slices_grid(
        image_t1=fixed,
        image_t2=registered_moving,
        mask_t1=mask_fixed,
        mask_t2=mask_moving,
        output_path=os.path.join(RESULTS_DIR, "fig_slices_grid.png"),
        y_center=SEED_FIXED[1],
    )

    print("   [4b] Visualisation 3D (fenêtre interactive VTK)...")
    display_comparison(
        fixed_path=FIXED_PATH,
        mask_t1_path=os.path.join(RESULTS_DIR, "mask_t1.nrrd"),
        mask_t2_path=os.path.join(RESULTS_DIR, "mask_t2.nrrd"),
    )


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Chargement des images...")
    fixed = load_image(FIXED_PATH)
    moving = load_image(MOVING_PATH)

    registered = run_registration(fixed, moving)
    mask_fixed, mask_moving = run_segmentation(fixed, registered)
    metrics = run_analysis(fixed, registered, mask_fixed, mask_moving)
    run_visualisation(fixed, registered, mask_fixed, mask_moving)


if __name__ == "__main__":
    main()
