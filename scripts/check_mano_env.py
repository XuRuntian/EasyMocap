"""Smoke-test MANO loading, CUDA shape fitting, detection and EGL rendering.

Run from the repository root: python scripts/check_mano_env.py
"""
import os

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import numpy as np
import torch

from easymocap.dataset import CONFIG
from easymocap.pyfitting import optimizeShape
from easymocap.smplmodel import load_model


def check_shape(side):
    model = load_model(model_type='mano' + side)
    target_params = model.init_params(nFrames=1)
    target_params['shapes'][0, :3] = [1.0, -0.5, 0.7]
    target = model(return_verts=False, return_tensor=False,
                   only_shape=True, **target_params)
    observations = np.concatenate([target, np.ones((*target.shape[:-1], 1))], axis=-1)
    observations = np.repeat(observations, 4, axis=0)
    initial = model.init_params(nFrames=1)
    edges = np.asarray(CONFIG['hand' + side]['kintree'])

    def length_error(params):
        joints = model(return_verts=False, return_tensor=False,
                       only_shape=True, **params)
        lengths = np.linalg.norm(joints[:, edges[:, 0]] - joints[:, edges[:, 1]], axis=-1)
        expected = np.linalg.norm(target[:, edges[:, 0]] - target[:, edges[:, 1]], axis=-1)
        return np.mean((lengths - expected) ** 2)

    before = length_error(initial)
    fitted = optimizeShape(model, initial, observations,
                           weight_loss={'s3d': 100., 'reg_shapes': 5e-5},
                           kintree=edges)
    after = length_error(fitted)
    assert np.isfinite(fitted['shapes']).all()
    assert after < before, (before, after)
    vertices = model(return_verts=True, return_tensor=False, **fitted)[0]
    assert vertices.shape == (778, 3)
    print(f'MANO {side}: vertices={vertices.shape}, bone MSE {before:.3g} -> {after:.3g}')
    return vertices, model.faces


def check_new_loader():
    from myeasymocap.io.model import MANOLoader

    loader = MANOLoader('config/model/manol.yml',
                        'models/manov1.2/MANO_LEFT.pkl',
                        'models/manov1.2/J_regressor_mano_LEFT.txt')
    params = loader.manomodel.init_params(nFrames=1, ret_tensor=True)
    joints = loader.forward(params)['keypoints']
    assert joints.shape == (1, 21, 3)
    assert torch.isfinite(joints).all()
    print('Multi-stage MANO loader: OK')


def check_detection():
    from myeasymocap.backbone.mediapipe.hand import MediaPipe

    detector = MediaPipe('models/mediapipe/hand_landmarker.task')
    try:
        result = detector(['check/000000.jpg'], [np.zeros((256, 256, 3), dtype=np.uint8)])
        assert result['keypoints'].shape == (1, 21, 3)
    finally:
        for instance in detector.detector.values():
            instance.close()
    print('MediaPipe blank-frame inference: OK')


def check_render(vertices, faces):
    import pyrender
    import trimesh

    vertices = vertices - vertices.mean(axis=0)
    scene = pyrender.Scene(ambient_light=[0.8, 0.8, 0.8])
    scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices, faces, process=False)))
    pose = np.eye(4)
    pose[2, 3] = 0.5
    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3), pose=pose)
    renderer = pyrender.OffscreenRenderer(256, 256)
    try:
        color, depth = renderer.render(scene)
        pixels = int(np.count_nonzero(depth))
        assert pixels > 0
        assert np.isfinite(color).all()
        print(f'EGL MANO rendering: OK ({pixels} foreground pixels)')
    finally:
        renderer.delete()


if __name__ == '__main__':
    assert torch.cuda.is_available(), 'CUDA is not available'
    print(f'PyTorch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}')
    vertices, faces = check_shape('l')
    check_shape('r')
    check_new_loader()
    check_detection()
    check_render(vertices, faces)
    print('All MANO environment checks passed.')
