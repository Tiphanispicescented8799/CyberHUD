import unittest
from unittest.mock import patch, Mock
import numpy as np
from cyberhud.geometry import Box
from cyberhud.detection import FaceDetector, Detection
from cyberhud.tracking import Tracker, tracking_status
from cyberhud.hud import HUD


class V2TrackingTests(unittest.TestCase):
    def setUp(self):
        self.box = Box(80,80,100,100)
        self.tracker = Tracker()

    def test_session_numbering(self):
        boxes = [self.box,Box(300,80,100,100),Box(500,80,100,100)]
        for tracker in (self.tracker,Tracker()):
            self.assertEqual([t.label for t in tracker.update(boxes,0)],
                             ['TARGET-01','TARGET-02','TARGET-03'])

    def test_state_sequence_and_short_miss(self):
        tr,b = self.tracker,self.box
        self.assertEqual(tr.update([b],0)[0].state,'ACQUIRING')
        self.assertEqual(tr.update([b],0.2)[0].state,'LOCKED')
        tr.update([b],0.5)
        self.assertEqual(tr.update([b],0.8)[0].state,'TRACKING')
        self.assertEqual(tr.update([],0.9)[0].state,'TRACKING')
        self.assertEqual(tr.update([b],1.0)[0].id,1)
        self.assertEqual(tr.update([],1.5)[0].state,'LOST')
        target = tr.update([b],1.6)[0]
        self.assertEqual((target.id,target.state),(1,'ACQUIRING'))
        self.assertEqual(tr.update([b],1.8)[0].state,'LOCKED')
        self.assertEqual(tr.update([],3.0)[0].state,'LOST')
        self.assertEqual(tr.update([],3.5),[])

    def test_single_detection_cannot_lock(self):
        tr = self.tracker
        tr.update([self.box],0)
        self.assertEqual(tr.update(None,0.3)[0].state,'ACQUIRING')
        self.assertEqual(tr.update(None,0.5)[0].state,'LOST')
        self.assertEqual(tr.update(None,1.7),[])

    def test_100_brief_misses_do_not_burn_ids(self):
        tr = self.tracker
        for i in range(100):
            targets = tr.update([self.box],i*0.2)
            tr.update([],i*0.2+0.1)
            self.assertEqual(targets[0].label,'TARGET-01')
        self.assertEqual(tr.next_id,2)

    def test_global_status_includes_retained_lost_targets(self):
        tr = self.tracker
        self.assertEqual(tracking_status([]),('SCANNING / NO TARGET','SCANNING',0))
        for timestamp,detections in [(0,[self.box]),(0.2,[self.box]),(0.8,[])]:
            targets = tr.update(detections,timestamp)
            headline,state,count = tracking_status(targets)
            self.assertEqual(headline,'TARGET TRACKING // ACTIVE')
            self.assertEqual(state,targets[0].state)
            self.assertEqual(count,1)
        self.assertEqual(tracking_status(tr.update([],2))[2],0)

    def test_mixed_faces_and_confidence(self):
        a,b = self.box,Box(400,80,100,100)
        tr = self.tracker
        tr.update([Detection(a,0.95),Detection(b,0.88)],0)
        tr.update([Detection(b,0.91),Detection(a,0.96)],0.2)
        tr.update([Detection(b,0.92)],0.5)
        targets = tr.update([Detection(b,0.93)],0.8)
        self.assertEqual([(t.id,t.state) for t in targets],[(1,'LOST'),(2,'TRACKING')])
        self.assertEqual(targets[0].confidence,0.96)
        self.assertEqual(tracking_status(targets),('TARGET TRACKING // ACTIVE','TRACKING',2))

    def test_hud_uses_consistent_status(self):
        targets = self.tracker.update([Detection(self.box,0.9)],0)
        for now in (0,0.6):
            targets = self.tracker.update(None,now)
            with patch('cyberhud.hud.text') as draw_text:
                HUD().draw(np.zeros((540,960,3),np.uint8),targets,now,21,'12:00:00')
            messages = [call.args[1] for call in draw_text.call_args_list]
            self.assertIn('TARGET TRACKING // ACTIVE',messages)
            self.assertFalse(any('SEARCHING' in m or 'COAST' in m for m in messages))


class V2DetectorTests(unittest.TestCase):
    def test_corrupt_model_fails_at_initialization(self):
        with patch('cyberhud.detection.cv2.FaceDetectorYN.create',side_effect=RuntimeError('bad model')):
            with self.assertRaisesRegex(RuntimeError,'initialization failed'):
                FaceDetector()

    def test_multiple_faces_clipped_and_scores_preserved(self):
        detector = FaceDetector()
        detector.classifier = Mock()
        detector.classifier.detect.return_value = (2,np.array([
            [-10,20,50,60]+[0]*10+[0.8], [200,40,60,70]+[0]*10+[0.95]]))
        result = detector.detect(np.zeros((240,320,3),np.uint8))
        self.assertEqual(len(result),2)
        self.assertEqual(result[0],Detection(Box(0,20,40,60),0.8))
        self.assertEqual(result[1].confidence,0.95)

    def test_model_loads_from_other_working_directory(self):
        from pathlib import Path
        import os
        old = Path.cwd()
        try:
            os.chdir(old/'.venv')
            detector = FaceDetector()
            self.assertEqual(detector.detect(np.zeros((180,320,3),np.uint8)),[])
        finally:
            os.chdir(old)
