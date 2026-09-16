import math
import vtk

def _read_surface(mask_path):
    reader = vtk.vtkNrrdReader()
    reader.SetFileName(mask_path)
    reader.Update()
    cast = vtk.vtkImageCast()
    cast.SetInputConnection(reader.GetOutputPort())
    cast.SetOutputScalarTypeToFloat()
    cast.Update()
    smooth = vtk.vtkImageGaussianSmooth()
    smooth.SetInputConnection(cast.GetOutputPort())
    smooth.SetStandardDeviation(1.5)
    smooth.Update()
    contour = vtk.vtkMarchingCubes()
    contour.SetInputConnection(smooth.GetOutputPort())
    contour.SetValue(0, 0.5)
    contour.Update()
    return contour


def display_comparison(fixed_path, mask_t1_path, mask_t2_path):
    """
    Visualisation 3D interactive.
      ESPACE      : alterner T1 (rouge) / T2 (bleu)
      Clic gauche : rotation libre (azimut Y + élévation X)
      ← →         : azimut
      ↑ ↓         : élévation
      S           : crâne on/off
      Molette     : zoom
    """
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.05, 0.05, 0.1)

    # ── Cerveau volume ────────────────────────────────────────────────────────
    brain_reader = vtk.vtkNrrdReader()
    brain_reader.SetFileName(fixed_path)
    brain_reader.Update()

    brain_mapper = vtk.vtkSmartVolumeMapper()
    brain_mapper.SetInputConnection(brain_reader.GetOutputPort())

    color_tf = vtk.vtkColorTransferFunction()
    color_tf.AddRGBPoint(0,   0.0, 0.0, 0.0)
    color_tf.AddRGBPoint(100, 0.3, 0.3, 0.3)
    color_tf.AddRGBPoint(400, 0.7, 0.7, 0.7)
    color_tf.AddRGBPoint(800, 1.0, 1.0, 1.0)

    opacity_tf = vtk.vtkPiecewiseFunction()
    opacity_tf.AddPoint(0,   0.00)
    opacity_tf.AddPoint(80,  0.00)
    opacity_tf.AddPoint(200, 0.03)
    opacity_tf.AddPoint(500, 0.07)
    opacity_tf.AddPoint(800, 0.10)

    brain_prop = vtk.vtkVolumeProperty()
    brain_prop.SetColor(color_tf)
    brain_prop.SetScalarOpacity(opacity_tf)
    brain_prop.ShadeOff()

    brain_volume = vtk.vtkVolume()
    brain_volume.SetMapper(brain_mapper)
    brain_volume.SetProperty(brain_prop)
    renderer.AddVolume(brain_volume)

    # ── Surfaces 3D tumorales ─────────────────────────────────────────────────
    c1 = _read_surface(mask_t1_path)
    pm1 = vtk.vtkPolyDataMapper()
    pm1.SetInputConnection(c1.GetOutputPort())
    pm1.ScalarVisibilityOff()
    surf_t1 = vtk.vtkActor()
    surf_t1.SetMapper(pm1)
    surf_t1.GetProperty().SetColor(1.0, 0.1, 0.1)
    surf_t1.GetProperty().SetSpecular(0.5)
    surf_t1.GetProperty().SetSpecularPower(40)
    renderer.AddActor(surf_t1)

    c2 = _read_surface(mask_t2_path)
    print(f"   Surface : {c2.GetOutput().GetNumberOfPoints()} points")
    pm2 = vtk.vtkPolyDataMapper()
    pm2.SetInputConnection(c2.GetOutputPort())
    pm2.ScalarVisibilityOff()
    surf_t2 = vtk.vtkActor()
    surf_t2.SetMapper(pm2)
    surf_t2.GetProperty().SetColor(0.1, 0.4, 1.0)
    surf_t2.GetProperty().SetSpecular(0.5)
    surf_t2.GetProperty().SetSpecularPower(40)
    surf_t2.SetVisibility(False)
    renderer.AddActor(surf_t2)

    # ── Légende ───────────────────────────────────────────────────────────────
    legend = vtk.vtkTextActor()
    legend.GetTextProperty().SetFontSize(15)
    legend.GetTextProperty().SetColor(0.9, 0.9, 0.9)
    legend.SetPosition(15, 15)
    CONTROLS = "ESPACE: T1/T2  |  Souris G: rotation  |  S: crâne  |  Molette: zoom"

    def update_legend():
        label = "[T1 — rouge]" if tumor_st[0] == 0 else "[T2 — bleu]"
        skull = "crâne ON" if skull_vis[0] else "crâne OFF"
        legend.SetInput(f"{label}  —  {skull}\n{CONTROLS}")

    renderer.AddActor2D(legend)

    # ── Fenêtre ───────────────────────────────────────────────────────────────
    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    window.SetWindowName("Evolution tumorale 3D")
    window.SetSize(1024, 800)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleUser())

    # ── Caméra sphérique ─────────────────────────────────────────────────────
    bds = brain_volume.GetBounds()
    cx = (bds[0] + bds[1]) / 2
    cy = (bds[2] + bds[3]) / 2
    cz = (bds[4] + bds[5]) / 2
    R_init = max(bds[1]-bds[0], bds[3]-bds[2], bds[5]-bds[4]) * 2.0

    azimuth   = [math.pi]
    elevation = [0.0]
    dist      = [R_init]
    tumor_st  = [0]
    skull_vis = [True]
    EL_MAX    = math.radians(85)

    def reposition_camera():
        az = azimuth[0]
        el = elevation[0]
        r  = dist[0]

        # Position sphérique
        px = cx + r * math.cos(el) * math.cos(az)
        py = cy - r * math.sin(el)
        pz = cz + r * math.cos(el) * math.sin(az)

        cam = renderer.GetActiveCamera()
        cam.SetPosition(px, py, pz)
        cam.SetFocalPoint(cx, cy, cz)

        # Vecteur de vue normalisé (focal → camera)
        lx = cx - px; ly = cy - py; lz = cz - pz
        ln = math.sqrt(lx*lx + ly*ly + lz*lz)
        lx /= ln; ly /= ln; lz /= ln

        # world_up = (0, -1, 0)  (Supérieur = -Y dans cet espace)
        # right = cross(look, world_up)
        rx = ly * 0.0 - lz * (-1.0)   # = lz
        ry = lz * 0.0 - lx *  0.0     # = 0
        rz = lx * (-1.0) - ly * 0.0   # = -lx
        rn = math.sqrt(rx*rx + ry*ry + rz*rz)
        if rn > 1e-6:
            rx /= rn; ry /= rn; rz /= rn

        # view_up = cross(right, look)
        ux = ry*lz - rz*ly
        uy = rz*lx - rx*lz
        uz = rx*ly - ry*lx
        cam.SetViewUp(ux, uy, uz)
        renderer.ResetCameraClippingRange()
    
    # clavier 
    def on_key(obj, event):
        key = interactor.GetKeySym()

        if key == 'space':
            tumor_st[0] = (tumor_st[0] + 1) % 2
            surf_t1.SetVisibility(tumor_st[0] == 0)
            surf_t2.SetVisibility(tumor_st[0] == 1)
            update_legend(); window.Render()


        elif key in ('s', 'S'):
            skull_vis[0] = not skull_vis[0]
            brain_volume.SetVisibility(skull_vis[0])
            update_legend(); window.Render()

    # souris
    mouse_down = [False]
    last_pos   = [0, 0]

    def on_left_down(obj, event):
        mouse_down[0] = True
        last_pos[0], last_pos[1] = interactor.GetEventPosition()

    def on_left_up(obj, event):
        mouse_down[0] = False

    def on_mouse_move(obj, event):
        if not mouse_down[0]:
            return
        x, y = interactor.GetEventPosition()
        dx = x - last_pos[0]
        dy = y - last_pos[1]
        last_pos[0], last_pos[1] = x, y
        azimuth[0]   -= math.radians(dx * 0.4)
        elevation[0]  = max(-EL_MAX, min(EL_MAX,
                            elevation[0] - math.radians(dy * 0.4)))
        reposition_camera(); window.Render()

    # zoom
    def on_scroll_forward(obj, event):
        dist[0] *= 0.9; reposition_camera(); window.Render()

    def on_scroll_backward(obj, event):
        dist[0] *= 1.1; reposition_camera(); window.Render()

    interactor.AddObserver('KeyPressEvent',           on_key)
    interactor.AddObserver('LeftButtonPressEvent',    on_left_down)
    interactor.AddObserver('LeftButtonReleaseEvent',  on_left_up)
    interactor.AddObserver('MouseMoveEvent',          on_mouse_move)
    interactor.AddObserver('MouseWheelForwardEvent',  on_scroll_forward)
    interactor.AddObserver('MouseWheelBackwardEvent', on_scroll_backward)

    update_legend()
    reposition_camera()
    window.Render()
    interactor.Start()
