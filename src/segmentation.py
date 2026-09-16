import itk
import numpy as np

Dimension = 3
PixelType = itk.F
ImageType = itk.Image[PixelType, Dimension]
_BinType = itk.Image[itk.SS, Dimension]
_LblType = itk.Image[itk.UL, Dimension]

def _otsu_threshold(arr):
    """Otsu 2 étapes : sépare d'abord cerveau/fond, puis tumeur/tissu brillant.
    Retenu comme meilleure méthode (cf. exploration/segmentation_exploration.py) :
    seuil calculé automatiquement, volume supérieur, validé visuellement."""
    def otsu(values):
        hist, bin_edges = np.histogram(values, bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        total = hist.sum()
        sum_total = (hist * bin_centers).sum()
        sum_bg, w_bg, best_var, thresh = 0.0, 0.0, 0.0, 0.0
        for h, bc in zip(hist, bin_centers):
            w_bg += h
            w_fg = total - w_bg
            if w_bg == 0 or w_fg == 0:
                continue
            sum_bg += h * bc
            mean_bg = sum_bg / w_bg
            mean_fg = (sum_total - sum_bg) / w_fg
            var = w_bg * w_fg * (mean_bg - mean_fg) ** 2
            if var > best_var:
                best_var = var
                thresh = bc
        return thresh

    thresh1 = otsu(arr[arr > 100])
    return otsu(arr[arr > thresh1]) * 0.85


def segment_tumor(image, seed):
    """
    Segmentation par seuillage Otsu 2 étapes + composante connexe.

    Seuil calculé automatiquement sans paramètre manuel.
    seed : (x, y, z) en coordonnées voxel ITK.
    """
    arr = itk.array_from_image(image)

    lower_threshold = _otsu_threshold(arr)

    bin_arr = (arr >= lower_threshold).astype(np.int16)
    bin_img = itk.image_from_array(bin_arr)
    bin_img.CopyInformation(image)

    cc = itk.ConnectedComponentImageFilter[_BinType, _LblType].New()
    cc.SetInput(bin_img)
    cc.Update()

    label_arr = itk.array_from_image(cc.GetOutput())
    seed_label = int(label_arr[seed[2], seed[1], seed[0]])

    if seed_label == 0:
        raise ValueError(
            f"Le seed {seed} est hors du masque (intensité {arr[seed[2],seed[1],seed[0]]:.0f} "
            f"< seuil Otsu {lower_threshold:.0f}). Ajuster le seed."
        )

    mask_arr = (label_arr == seed_label).astype(np.uint8)
    mask = itk.image_from_array(mask_arr)
    mask.CopyInformation(image)
    mask.DisconnectPipeline()
    return mask


def count_voxels(mask):
    arr = itk.array_from_image(mask)
    return int((arr > 0).sum())


def voxel_volume_mm3(mask):
    spacing = mask.GetSpacing()
    return float(spacing[0] * spacing[1] * spacing[2])


def save_mask(mask, filename):
    itk.imwrite(mask, filename)
