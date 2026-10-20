"""Offline regression tests for project isolation, evidence integrity and portability."""
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ds_context',ROOT/'scripts/ds_context.py')
ctx=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctx)

class ContextTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        manifest=ctx.read_json(ROOT/'context-release.json')
        for item in manifest['files']:
            rel=item['path'];p=self.root/rel;p.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/rel,p)
        shutil.copy2(ROOT/'context-release.json',self.root/'context-release.json')

    def test_portable_and_not_authorization(self):
        result=ctx.load_context(self.root,'flip_anything')
        self.assertFalse(result['hardware_authorization'])
        self.assertEqual(result['current_state'],'unknown_until_fresh_observation')
        tcp=result['calibration']['operational_grasp_tcp']
        self.assertNotIn('load_lift_test_performed',tcp['validation'])
        self.assertAlmostEqual(tcp['operational_grasp_tcp']['controller_tcp_from_operational_grasp_reference'][2][3],-0.01699214606154837)
        self.assertFalse(result['experiences'][0]['autonomous_success'])

    def test_default_context_is_compact_but_full_data_available(self):
        compact=ctx.load_context(self.root,'flip_anything')
        full=ctx.load_context(self.root,'flip_anything',full=True)
        self.assertLess(len(json.dumps(compact,ensure_ascii=False)),20000)
        self.assertNotIn('samples',compact['calibration']['opening_dependent_grasp_model'])
        self.assertIn('samples',full['calibration']['opening_dependent_grasp_model'])
        self.assertEqual(compact['calibration']['hand_eye_profile']['tool_from_camera'],full['calibration']['hand_eye_profile']['tool_from_camera'])

    def test_modified_measurement_rejected(self):
        env=ctx.read_json(self.root/'environments/xarm6_lab/20261020/environment.json')
        (self.root/env['components']['operational_grasp_tcp']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'Changed context file'):ctx.load_context(self.root,'flip_anything')

    def test_missing_evidence_rejected(self):
        (self.root/'knowledge/flip_anything/assisted_can_inversion_20261014.json').unlink()
        with self.assertRaisesRegex(ValueError,'Missing context file'):ctx.load_context(self.root,'flip_anything')

    def test_unknown_project_does_not_fallback(self):
        with self.assertRaisesRegex(ValueError,'Untracked context reference'):ctx.load_context(self.root,'another_project')

    def test_traversal_and_absolute_paths_rejected(self):
        for name in ['../outside','C:/outside','/outside','a\\b']:
            with self.subTest(name=name),self.assertRaises(ValueError):ctx.path_in(self.root,name)
        with self.assertRaises(ValueError):ctx.load_context(self.root,'../flip_anything')

    def test_project_scopes_are_checked(self):
        rel='knowledge/flip_anything/assisted_can_inversion_20261014.json'
        card=ctx.read_json(self.root/rel);card['project_id']='another_project'
        (self.root/rel).write_text(json.dumps(card))
        manifest=ctx.read_json(self.root/'context-release.json')
        for item in manifest['files']:
            if item['path']==rel:item['sha256']=ctx.hashlib.sha256((self.root/rel).read_bytes()).hexdigest()
        (self.root/'context-release.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'Experience scope mismatch'):ctx.load_context(self.root,'flip_anything')

    def test_mutable_release_read_fresh(self):
        p=self.root/'devices/calibration_release.json';p.parent.mkdir()
        p.write_text('{"status":"draft"}')
        self.assertEqual(ctx.load_context(self.root,'flip_anything')['pipeline_calibration_release_status'],'draft')
        p.write_text('{"status":"revoked"}')
        self.assertEqual(ctx.load_context(self.root,'flip_anything')['pipeline_calibration_release_status'],'revoked')

if __name__=='__main__':unittest.main()
