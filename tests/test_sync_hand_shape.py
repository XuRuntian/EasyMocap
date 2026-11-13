import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('calibrate_hand_shape', Path(__file__).resolve().parents[1]/'scripts/calibrate_hand_shape.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SyncShapeTests(unittest.TestCase):
    def test_rectification_identity_and_validation(self):
        intr = dict(width=8, height=6, fx=20., fy=20., ppx=4., ppy=3., model='distortion.inverse_brown_conrady', coeffs=[0.]*5)
        x, y = module.rectification_maps(intr)
        yy, xx = np.mgrid[:6, :8]
        np.testing.assert_allclose(x, xx)
        np.testing.assert_allclose(y, yy)
        with self.assertRaises(ValueError):
            module.rectification_maps({**intr, 'model': 'unknown'})

    def test_triangulation_outlier_missing_and_behind_camera(self):
        k = np.diag([500., 500., 1.])
        matrices = np.stack([k @ np.column_stack([np.eye(3), [-i*.1, 0, 0]]) for i in range(4)])
        point = np.array([.03, .04, 1., 1.])
        q = matrices @ point
        obs = np.zeros((4, 2, 21, 3))
        obs[:, 0, :, :2] = (q[:, :2]/q[:, 2:])[:, None]
        obs[:, 0, :, 2] = 1
        obs[3, 0, :, :2] += 100
        joints, valid, inliers = module.triangulate(obs, matrices, 2.)
        self.assertTrue(valid[:, :].any())
        self.assertTrue(valid[0].all())
        self.assertFalse(valid[1].any())
        np.testing.assert_allclose(joints[0], np.tile(point[:3], (21, 1)), atol=1e-5)
        self.assertFalse(inliers[3].any())
        obs[2:, :, :, 2] = 0
        self.assertFalse(module.triangulate(obs, matrices, 2.)[1].any())
        point[2] = -1.
        q = matrices @ point
        obs[:, 0, :, :2] = (q[:, :2]/q[:, 2:])[:, None]
        obs[:, 0, :, 2] = 1
        self.assertFalse(module.triangulate(obs, matrices, 2.)[1].any())

    def test_clip_without_ego_or_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rgbd = root/'rgbd'
            rgbd.mkdir()
            (root/'calibration').mkdir()
            module.write(rgbd/'export_manifest.json', dict(schema='sync_lab.rgbd_images.v1', status='complete'))
            cameras, transforms = {}, {}
            intr = dict(width=8, height=6, fx=20., fy=20., ppx=4., ppy=3., model='none', coeffs=[0.]*5)
            for i in range(3):
                name, serial = f'cam{i:02d}', str(i)
                (rgbd/name).mkdir()
                (rgbd/name/'color.jpg').write_bytes(b'test existence only')
                module.write(rgbd/name/'calibration.json', dict(id=name, serial=serial, original_calibration=dict(intrinsics=dict(color=intr))))
                cameras[name] = dict(serial=serial, color_path=f'{name}/color.jpg', color_source=dict(timestamp_ms=0))
                t = np.eye(4)
                t[0, 3] = i*.1
                transforms[serial] = t.tolist()
            module.write(root/'calibration/extrinsics.json', dict(master_serial='0', translation_unit='meter', convention='p_master = T_master_from_camera @ p_camera (homogeneous column vectors)', T_master_from_camera=transforms))
            row = dict(frame_id=7, cameras=cameras)
            (rgbd/'frames.jsonl').write_text(json.dumps(row)+'\n')
            loaded = module.load_clip(root)
            self.assertEqual(loaded[1], ['cam00', 'cam01', 'cam02'])
            self.assertAlmostEqual(loaded[3][1, 0, 3], -.1)
            cameras['cam00']['color_path'] = '../outside.jpg'
            (rgbd/'frames.jsonl').write_text(json.dumps(row)+'\n')
            with self.assertRaises(ValueError):
                module.load_clip(root)


if __name__ == '__main__':
    unittest.main()
