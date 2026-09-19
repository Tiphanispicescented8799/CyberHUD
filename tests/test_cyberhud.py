import unittest
from unittest.mock import Mock, patch
import numpy as np
import cv2
from cyberhud.geometry import Box, iou, bracket_segments
from cyberhud.tracking import Tracker, tracking_status
from cyberhud.effects import scan_y
from cyberhud.camera import Camera
from cyberhud.detection import FaceDetector, Detection
from cyberhud.hud import HUD
from cyberhud.app import run, main


class GeometryTests(unittest.TestCase):
    def test_center_iou_and_clipping(self):
        box = Box(10,20,40,60)
        self.assertEqual(box.center,(30,50))
        self.assertEqual(iou(box,box),1)
        self.assertEqual(iou(box,Box(100,100,1,1)),0)
        self.assertEqual(Box(-10,-20,200,200).clipped(100,80),(0,0,99,79))

    def test_brackets_stay_inside(self):
        segments = bracket_segments((0,0,10,8),100)
        self.assertEqual(len(segments),8)
        for segment in segments:
            for x,y in segment:
                self.assertTrue(0 <= x <= 10 and 0 <= y <= 8)

    def test_scan_bounds_and_period(self):
        for t in np.linspace(0,10,100):
            self.assertTrue(10 <= scan_y(10,110,t) <= 110)
        self.assertEqual(scan_y(10,110,0),10)
        self.assertEqual(scan_y(10,110,1.2),110)
        self.assertEqual(scan_y(10,110,2.4),10)


class TrackingTests(unittest.TestCase):
    def test_ids_survive_reordered_detections(self):
        tr = Tracker()
        a,b = Box(0,0,50,50),Box(200,0,50,50)
        tr.update([a,b],0)
        targets = tr.update([b,a],0.1)
        self.assertEqual([(t.id,t.measured) for t in targets],[(1,a),(2,b)])

    def test_smoothing_and_skipped_detection(self):
        tr = Tracker()
        tr.update([Box(0,0,100,100)],0)
        target = tr.update([Box(20,0,100,100)],0.03)[0]
        self.assertTrue(0 < target.box.x < 20)
        old = target.box.x
        self.assertGreater(tr.update(None,0.06)[0].box.x,old)
        self.assertEqual(target.last_seen,0.03)

    def test_expiry_and_new_id(self):
        tr = Tracker(ttl=0.5)
        a = Box(0,0,50,50)
        tr.update([a],0)
        self.assertFalse(tr.update([],0.1)[0].visible)
        self.assertEqual(tr.update([a],0.2)[0].id,1)
        self.assertEqual(tr.update(None,0.8),[])
        self.assertEqual(tr.update([a],0.9)[0].label,'TARGET-02')

    def test_one_to_one_assignment(self):
        tr = Tracker()
        tr.update([Box(0,0,80,80),Box(20,0,80,80)],0)
        targets = tr.update([Box(10,0,80,80)],0.1)
        self.assertEqual(sum(t.visible for t in targets),1)


class DetectionRenderingTests(unittest.TestCase):
    def test_real_detector_blank_frame(self):
        self.assertEqual(FaceDetector().detect(np.zeros((240,320,3),np.uint8)),[])

    def test_invalid_model(self):
        with self.assertRaisesRegex(RuntimeError,'initialization failed'):
            FaceDetector(model_path='missing-cyberhud-model.xml')

    def test_scaled_coordinates(self):
        detector = FaceDetector(width=480)
        detector.classifier = Mock()
        detector.classifier.detect.return_value = (1,np.array([[10,20,30,40]+[0]*10+[0.9]]))
        self.assertEqual(detector.detect(np.zeros((540,960,3),np.uint8)),
                         [Detection(Box(20,40,60,80),0.9)])

    def test_render_edges_without_camera_or_window(self):
        tracker = Tracker()
        targets = tracker.update([Box(-10,-10,100,100),Box(900,490,80,80)],0)
        for size in ((540,960),(240,320)):
            frame = np.zeros((*size,3),np.uint8)
            output = HUD().draw(frame,targets,1,30,'2026-09-18 12:00:00')
            self.assertEqual(output.shape,frame.shape)
            self.assertGreater(int(output.sum()),0)


class LifecycleTests(unittest.TestCase):
    def camera(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (True,np.zeros((240,320,3),np.uint8))
        return Camera(factory=Mock(return_value=capture)),capture

    def test_unavailable_camera_releases(self):
        capture = Mock()
        capture.isOpened.return_value = False
        with self.assertRaisesRegex(RuntimeError,'unavailable'):
            Camera(factory=Mock(return_value=capture)).open()
        self.assertGreaterEqual(capture.release.call_count,1)

    def test_configuration_failure_releases(self):
        camera,capture = self.camera()
        capture.set.side_effect = RuntimeError('settings failed')
        with self.assertRaises(RuntimeError):
            camera.open()
        capture.release.assert_called_once()

    def test_context_exception_cleanup(self):
        camera,capture = self.camera()
        with self.assertRaises(ValueError):
            with camera:
                raise ValueError('test')
        camera.close()
        capture.release.assert_called_once()

    def exercise_loop(self, mode):
        camera,capture = self.camera()
        detector = Mock()
        detector.detect.return_value = []
        with patch.multiple(cv2,namedWindow=Mock(),imshow=Mock(),
                            waitKey=Mock(return_value=ord('q') if mode=='quit' else -1),
                            getWindowProperty=Mock(return_value=0 if mode=='close' else 1),
                            destroyAllWindows=Mock()) as _:
            if mode=='drop':
                capture.read.return_value = (False,None)
                with self.assertRaisesRegex(RuntimeError,'30 retries'):
                    run(camera,detector)
            elif mode=='error':
                detector.detect.side_effect = RuntimeError('detection failed')
                with self.assertRaises(RuntimeError):
                    run(camera,detector)
            else:
                run(camera,detector)
            cv2.destroyAllWindows.assert_called_once()
        capture.release.assert_called_once()

    def test_quit_cleanup(self): self.exercise_loop('quit')
    def test_window_close_cleanup(self): self.exercise_loop('close')
    def test_dropped_frame_cleanup(self): self.exercise_loop('drop')
    def test_detector_error_cleanup(self): self.exercise_loop('error')

    def test_detector_failure_before_camera(self):
        with patch('cyberhud.app.FaceDetector',side_effect=RuntimeError('failed')), \
             patch('cyberhud.app.Camera') as camera:
            self.assertEqual(main([]),1)
            camera.assert_not_called()

    def test_mirror_and_transient_drop_recovery(self):
        camera,capture = self.camera()
        frame = np.zeros((240,320,3),np.uint8)
        frame[:,0] = 255
        capture.read.side_effect = [(False,None),(True,frame)]
        detector = Mock()
        detector.detect.return_value = []
        with patch.multiple(cv2,namedWindow=Mock(),imshow=Mock(),
                            waitKey=Mock(side_effect=[-1,ord('q')]),
                            getWindowProperty=Mock(return_value=1),destroyAllWindows=Mock()):
            run(camera,detector,hud=Mock())
        detected_frame = detector.detect.call_args.args[0]
        np.testing.assert_array_equal(detected_frame[:,-1],frame[:,0])
        np.testing.assert_array_equal(detected_frame[:,0],frame[:,-1])
        capture.release.assert_called_once()

    def test_missing_dependency_message(self):
        import builtins
        import main as entry
        original_import = builtins.__import__

        def missing(name, *args, **kwargs):
            if name == 'cyberhud.app':
                raise ImportError('cv2 unavailable')
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__',side_effect=missing), \
             patch('sys.stderr') as stderr:
            self.assertEqual(entry.main(),1)
            self.assertTrue(stderr.write.called)


if __name__ == '__main__':
    unittest.main()
