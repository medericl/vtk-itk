import itk
import numpy as np


PixelType = itk.F
Dimension = 3
ImageType = itk.Image[PixelType, Dimension]


def load_image(filename):
    reader = itk.ImageFileReader[itk.Image[itk.US, Dimension]].New()
    reader.SetFileName(filename)
    reader.Update()

    caster = itk.CastImageFilter[
        itk.Image[itk.US, Dimension],
        ImageType
    ].New()
    caster.SetInput(reader.GetOutput())
    caster.Update()

    image = caster.GetOutput()
    image.DisconnectPipeline()
    return image


def save_image(image, filename):
    cast_back = itk.CastImageFilter[
        ImageType,
        itk.Image[itk.US, Dimension]
    ].New()
    cast_back.SetInput(image)
    cast_back.Update()

    writer = itk.ImageFileWriter[itk.Image[itk.US, Dimension]].New()
    writer.SetFileName(filename)
    writer.SetInput(cast_back.GetOutput())
    writer.Update()


class OptimizerObserver:
    def __init__(self, optimizer, metric):
        self.optimizer = optimizer
        self.metric = metric
        self.values = []
        self.optimizer.AddObserver(itk.IterationEvent(), self)

    def __call__(self, *args):
        self.values.append(self.metric.GetValue())


def initialize_transform(fixed, moving, transform):
    if isinstance(transform, itk.TranslationTransform[itk.D, Dimension]):
        fixed_center, moving_center = _compute_centers_of_mass(fixed, moving)

        offset = itk.Vector[itk.D, Dimension]()
        for i in range(Dimension):
            offset[i] = fixed_center[i] - moving_center[i]
        transform.SetOffset(offset)

    elif isinstance(transform, itk.AffineTransform[itk.D, Dimension]):
        fixed_center, moving_center = _compute_centers_of_mass(fixed, moving)

        center = itk.Point[itk.D, Dimension]()
        translation = itk.Vector[itk.D, Dimension]()
        for i in range(Dimension):
            center[i] = fixed_center[i]
            translation[i] = fixed_center[i] - moving_center[i]

        transform.SetIdentity()
        transform.SetCenter(center)
        transform.SetTranslation(translation)

    else:
        initializer = itk.CenteredTransformInitializer[
            type(transform), ImageType, ImageType
        ].New()
        initializer.SetTransform(transform)
        initializer.SetFixedImage(fixed)
        initializer.SetMovingImage(moving)
        initializer.MomentsOn()
        initializer.InitializeTransform()

    return transform


def _compute_centers_of_mass(fixed, moving):
    fixed_moments = itk.ImageMomentsCalculator[type(fixed)].New()
    fixed_moments.SetImage(fixed)
    fixed_moments.Compute()

    moving_moments = itk.ImageMomentsCalculator[type(moving)].New()
    moving_moments.SetImage(moving)
    moving_moments.Compute()

    return fixed_moments.GetCenterOfGravity(), moving_moments.GetCenterOfGravity()


def register(fixed, moving, transform, learning_rate=1.0,
             min_step=0.0001, n_iterations=500, n_bins=50,
             sampling_percentage=0.20, verbose=True,
             relaxation_factor=0.7):
    metric = itk.MattesMutualInformationImageToImageMetricv4[
        ImageType, ImageType
    ].New()
    metric.SetNumberOfHistogramBins(n_bins)
    metric.SetUseMovingImageGradientFilter(False)
    metric.SetUseFixedImageGradientFilter(False)

    optimizer = itk.RegularStepGradientDescentOptimizerv4[itk.D].New()
    optimizer.SetLearningRate(learning_rate)
    optimizer.SetMinimumStepLength(min_step)
    optimizer.SetNumberOfIterations(n_iterations)
    optimizer.SetRelaxationFactor(relaxation_factor)

    n_params = transform.GetNumberOfParameters()
    scales = itk.OptimizerParameters[itk.D](n_params)

    if isinstance(transform, itk.TranslationTransform[itk.D, Dimension]):
        for i in range(n_params):
            scales[i] = 1.0

    elif n_params == 6:
        translation_scale = 1.0 / 1000.0
        for i in range(3):
            scales[i] = 1.0
        for i in range(3, 6):
            scales[i] = translation_scale

    else:
        translation_scale = 1.0 / 1000.0
        for i in range(n_params - 3):
            scales[i] = 1.0
        for i in range(n_params - 3, n_params):
            scales[i] = translation_scale

    optimizer.SetScales(scales)
    optimizer.SetReturnBestParametersAndValue(True)

    registration = itk.ImageRegistrationMethodv4[
        ImageType, ImageType
    ].New()
    registration.SetFixedImage(fixed)
    registration.SetMovingImage(moving)
    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)
    registration.SetInitialTransform(transform)
    registration.InPlaceOn()

    registration.SetMetricSamplingStrategy(
        itk.ImageRegistrationMethodv4Enums.MetricSamplingStrategy_RANDOM
    )
    registration.SetMetricSamplingPercentage(sampling_percentage)

    n_levels = 3
    shrink_factors = itk.Array[itk.F](n_levels)
    shrink_factors[0] = 4
    shrink_factors[1] = 2
    shrink_factors[2] = 1

    smoothing_sigmas = itk.Array[itk.F](n_levels)
    smoothing_sigmas[0] = 2
    smoothing_sigmas[1] = 1
    smoothing_sigmas[2] = 0

    registration.SetNumberOfLevels(n_levels)
    registration.SetShrinkFactorsPerLevel(shrink_factors)
    registration.SetSmoothingSigmasPerLevel(smoothing_sigmas)

    observer = OptimizerObserver(optimizer, metric)

    if verbose:
        print(f"--- Démarrage du recalage : {type(transform).__name__} ---")

    registration.Update()

    final_metric_value = optimizer.GetValue()
    if verbose:
        print(f"Itérations effectuées : {optimizer.GetCurrentIteration()}")
        print(f"Valeur finale de la métrique (Mattes MI, négative) : "
              f"{final_metric_value:.6f}")
        print(f"Raison d'arrêt : {optimizer.GetStopConditionDescription()}")

    return registration.GetTransform(), observer, final_metric_value


def build_transform(kind):
    if kind == "translation":
        return itk.TranslationTransform[itk.D, Dimension].New()
    elif kind == "rigid":
        return itk.VersorRigid3DTransform[itk.D].New()
    elif kind == "affine":
        return itk.AffineTransform[itk.D, Dimension].New()
    else:
        raise ValueError(f"Type de transformation inconnu : {kind}")


def resample(fixed, moving, transform, default_value=0):
    resampler = itk.ResampleImageFilter[ImageType, ImageType].New()
    resampler.SetInput(moving)
    resampler.SetTransform(transform)
    resampler.SetUseReferenceImage(True)
    resampler.SetReferenceImage(fixed)
    resampler.SetDefaultPixelValue(default_value)
    resampler.Update()
    return resampler.GetOutput()


def compute_quality_metrics(fixed, registered_moving):
    fixed_np = itk.array_from_image(fixed).astype(np.float64)
    moving_np = itk.array_from_image(registered_moving).astype(np.float64)

    mask = moving_np > 0

    diff = fixed_np[mask] - moving_np[mask]
    mse = float(np.mean(diff ** 2))

    f = fixed_np[mask] - fixed_np[mask].mean()
    m = moving_np[mask] - moving_np[mask].mean()
    denom = np.sqrt(np.sum(f ** 2) * np.sum(m ** 2))
    ncc = float(np.sum(f * m) / denom) if denom != 0 else float("nan")

    return {"MSE": mse, "NCC": ncc}