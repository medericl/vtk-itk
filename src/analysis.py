import itk


def compare_tumors(image_t1, image_t2, mask_t1, mask_t2):
    arr1 = itk.array_from_image(image_t1).astype(float)
    arr2 = itk.array_from_image(image_t2).astype(float)
    m1   = itk.array_from_image(mask_t1).astype(bool)
    m2   = itk.array_from_image(mask_t2).astype(bool)

    spacing = image_t1.GetSpacing()
    vox_vol = float(spacing[0] * spacing[1] * spacing[2])

    # --- Volumes ---
    n1 = int(m1.sum()); n2 = int(m2.sum())
    vol1 = n1 * vox_vol; vol2 = n2 * vox_vol

    # --- Intensités ---
    mean1 = float(arr1[m1].mean()) if n1 > 0 else 0.0
    mean2 = float(arr2[m2].mean()) if n2 > 0 else 0.0

    # --- Dice ---
    intersection = int((m1 & m2).sum())
    dice = (2 * intersection) / (n1 + n2) if (n1 + n2) > 0 else 0.0

    return {
        "volume_t1_mm3":     vol1,
        "volume_t2_mm3":     vol2,
        "mean_intensity_t1": mean1,
        "mean_intensity_t2": mean2,
        "dice":              dice,
    }


def print_report(metrics):
    print("\n  --- Analyse de l'évolution tumorale ---")
    print(f"  Volume T1         : {metrics['volume_t1_mm3']:.0f} mm³  ({metrics['volume_t1_mm3']/1000:.2f} cm³)")
    print(f"  Volume T2         : {metrics['volume_t2_mm3']:.0f} mm³  ({metrics['volume_t2_mm3']/1000:.2f} cm³)")
    print(f"  Intensité moy T1  : {metrics['mean_intensity_t1']:.1f}")
    print(f"  Intensité moy T2  : {metrics['mean_intensity_t2']:.1f}")
    print(f"  Dice              : {metrics['dice']:.3f}  (0=aucun chevauchement, 1=identiques)")
