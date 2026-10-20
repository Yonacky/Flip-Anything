"""Task-scoped calibration approval; all fixtures are synthetic, no hardware."""
import hashlib
import unittest
from pathlib import Path
from flip_pipeline.storage import read, write, evidence
from flip_pipeline.calibration import profile_content_sha256
from flip_pipeline.calibration_reuse import reuse_calibration
from flip_pipeline.workflow import new_task, status
from pipeline_tests import test_calibration_reuse as reuse_fixtures

class TaskScopedCalibrationTests(unittest.TestCase):
    def setUp(self):
        reuse_fixtures.ReuseTests.setUp(self)
        contract=read(Path(__file__).resolve().parents[1]/'devices/task_profiles/supervised_rigid_flip.json')
        write(self.root/'devices/task_profiles/supervised_rigid_flip.json',contract)
        profile=read(self.root/'devices/profile.json')
        profile['review'].pop('independent_physical_check_passed')
        profile['review'].pop('physical_check_evidence_ref')
        profile['review'].update(basis='operator_accepted_measured_geometry',operator_accepted=True,
                                task_profiles=['supervised_rigid_flip'],operator_acceptance_ref='acceptance.json')
        write(self.root/'devices/profile.json',profile)
        write(self.root/'acceptance.json',{'source':'operator_declaration','accepted':True,
            'task_profiles':['supervised_rigid_flip'],'installation_id':profile['installation']['installation_id'],
            'profile_sha256':profile_content_sha256(profile)})
        self.task=new_task(self.root,'scoped-object',profile['installation']['installation_id'],profile='supervised_rigid_flip')
        self.refresh()

    def refresh(self,omit_acceptance=False):
        paths=['devices/installation.json','devices/profile.json','report.json','measurement.json']
        if not omit_acceptance:paths.append('acceptance.json')
        index=read(self.root/'devices/calibration_release.json')
        index.update(task_profiles=['supervised_rigid_flip'],evidence=evidence(self.root,paths))
        write(self.root/'devices/calibration_release.json',index)
        write(self.root/'source-release.json',{'release':'synthetic-test','sha256':{'devices/calibration_release.json':hashlib.sha256((self.root/'devices/calibration_release.json').read_bytes()).hexdigest()}})

    def test_accepted_scope_reuses_without_extra_physical_test(self):
        self.assertEqual(reuse_calibration(self.root,self.task)['stage'],'model')
        self.assertTrue(reuse_calibration(self.root,self.task)['already_attached'])

    def test_other_task_profile_not_silently_accepted(self):
        task=new_task(self.root,'other','synthetic-installation')
        with self.assertRaisesRegex(ValueError,'task profile'):reuse_calibration(self.root,task)

    def test_acceptance_must_be_hashed(self):
        self.refresh(omit_acceptance=True)
        with self.assertRaisesRegex(ValueError,'hashed evidence'):reuse_calibration(self.root,self.task)
        self.assertEqual(status(self.root,self.task)['stage'],'calibration')

    def test_acceptance_must_match_installation(self):
        p=self.root/'acceptance.json';a=read(p);a['installation_id']='different';write(p,a);self.refresh()
        with self.assertRaisesRegex(ValueError,'does not match'):reuse_calibration(self.root,self.task)

    def test_acceptance_does_not_replace_numerical_checks(self):
        p=self.root/'devices/profile.json';profile=read(p);profile['tool_from_camera'][0][3]+=0.1
        profile['review']['profile_sha256']=profile_content_sha256(profile);write(p,profile)
        a=read(self.root/'acceptance.json');a['profile_sha256']=profile_content_sha256(profile);write(self.root/'acceptance.json',a);self.refresh()
        with self.assertRaisesRegex(ValueError,'hand-eye evidence'):reuse_calibration(self.root,self.task)

if __name__=='__main__':unittest.main()
