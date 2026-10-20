"""Calibration reuse integration tests; synthetic fixtures, no hardware."""
import hashlib,tempfile,unittest
from pathlib import Path
from flip_pipeline.storage import write,read,evidence
from flip_pipeline.workflow import new_task,REQUIRED,ALLOWED_METHODS,status
from flip_pipeline.calibration_reuse import reuse_calibration
from pipeline_tests.test_calibration import profile_fixture,approve

class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        profile,installation=profile_fixture();installation['status']='verified';approve(profile)
        write(self.root/'devices/installation.json',installation);write(self.root/'devices/profile.json',profile)
        write(self.root/'measurement.json',{'synthetic':True})
        checks={k:{'result':'pass','method':next(iter(ALLOWED_METHODS[k])),'evidence':['measurement.json'],'detail':'synthetic fixture only'} for k in REQUIRED['calibration']}
        write(self.root/'report.json',{'source':'physical_evidence','checks':checks,'profile_path':'devices/profile.json','installation_path':'devices/installation.json'})
        refs=evidence(self.root,['devices/installation.json','devices/profile.json','report.json','measurement.json'])
        write(self.root/'devices/calibration_release.json',{'status':'verified','installation_id':installation['installation_id'],'installation_path':'devices/installation.json','report_path':'report.json','evidence':refs})
        write(self.root/'source-release.json',{'release':'synthetic-test','sha256':{'devices/calibration_release.json':hashlib.sha256((self.root/'devices/calibration_release.json').read_bytes()).hexdigest()}})
        self.task=new_task(self.root,'synthetic-object',installation['installation_id'])
    def test_reuse_advances_to_model_and_is_idempotent(self):
        self.assertEqual(reuse_calibration(self.root,self.task)['stage'],'model')
        count=len(read(self.task/'state.json')['history'])
        self.assertTrue(reuse_calibration(self.root,self.task)['already_attached'])
        self.assertEqual(len(read(self.task/'state.json')['history']),count)
    def test_modified_evidence_does_not_advance(self):
        write(self.root/'measurement.json',{'tampered':True})
        with self.assertRaises(ValueError):reuse_calibration(self.root,self.task)
        self.assertEqual(status(self.root,self.task)['stage'],'calibration')
    def test_wrong_installation_does_not_advance(self):
        task=new_task(self.root,'other-object','other-installation')
        with self.assertRaisesRegex(ValueError,'installation'):reuse_calibration(self.root,task)
        self.assertEqual(status(self.root,task)['stage'],'calibration')

    def test_draft_release_is_refused_before_stage_changes(self):
        path=self.root/'devices/calibration_release.json';index=read(path);index['status']='draft';write(path,index)
        write(self.root/'source-release.json',{'release':'synthetic-test','sha256':{'devices/calibration_release.json':hashlib.sha256(path.read_bytes()).hexdigest()}})
        with self.assertRaisesRegex(ValueError,'verified calibration'):reuse_calibration(self.root,self.task)
        self.assertEqual(status(self.root,self.task)['stage'],'calibration')
