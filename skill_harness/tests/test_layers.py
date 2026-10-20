import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from skill_harness.catalog import load_catalog,resolve_plan,resolve_capability,validate_catalog
from skill_harness.runtime import new_episode,read,write,episode_capability,run_skill,assess,digest

class LayerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.ep=self.root/'episode'
        self.state=new_episode(self.ep,'test_run',plan_id='empty_can_inversion')
    def tearDown(self):self.tmp.cleanup()
    def facts(self,name,ref,revision):
        return {name:{'value':True,'revision':revision,'result_sha256':digest(ref),
                     'source':'perception','method':'test fixture','artifacts':[str(ref)]}}
    def test_goal_is_result_contract_not_action_route(self):
        layers=load_catalog();goal=layers['goals']['flip_object']
        self.assertNotIn('action',goal);self.assertNotIn('capabilities',goal)
        class NeverDispatch:
            def dispatch(self,*args):raise AssertionError('Goal must not dispatch hardware')
        with self.assertRaises(ValueError):
            run_skill(self.ep,'flip_object',{'action':'move'},{},None,NeverDispatch(),True)
        invalid=copy.deepcopy(layers);invalid['goals']['flip_object']['capabilities']=['grasp']
        with self.assertRaises(ValueError):validate_catalog(invalid)
    def test_specialization_preserves_constraints_and_reuses_capabilities(self):
        layers=load_catalog();base=resolve_plan('pick_invert_place',layers);can=resolve_plan('empty_can_inversion',layers)
        self.assertEqual(base['capabilities'],can['capabilities'])
        for group,requirements in base['applicability'].items():
            self.assertTrue(set(requirements).issubset(can['applicability'][group]))
        self.assertIn('空容器',can['applicability']['object'])
        self.assertEqual(can['goal'],'flip_object')
        self.assertNotEqual(base['goal_binding'],can['goal_binding'])
    def test_cycles_and_dangling_capabilities_rejected(self):
        layers=load_catalog();bad=copy.deepcopy(layers)
        bad['plans']['pick_invert_place']['specializes']='empty_can_inversion'
        with self.assertRaises(ValueError):validate_catalog(bad)
        bad=copy.deepcopy(layers);bad['plans']['pick_invert_place']['capabilities']=['nonexistent']
        with self.assertRaises(ValueError):validate_catalog(bad)
    def test_local_inversion_does_not_complete_task(self):
        ref=self.root/'result.json';write(ref,{'ok':True,'state':{'gripper':{'gFLT':0,'gOBJ':2}}})
        self.state.update(phase='awaiting_assessment',pending='invert_held_object',last_result=str(ref),backend='mock',revision=1)
        write(self.ep/'state.json',self.state)
        result=assess(self.ep,self.facts('inverted_held',ref,1))
        self.assertEqual(result['phase'],'ready');self.assertFalse((self.ep/'outcome.json').exists())
    def test_old_names_resolve_without_shadowing_goal(self):
        self.assertEqual(resolve_capability('flip')['id'],'invert_held_object')
        self.assertEqual(resolve_capability('verify')['id'],'verify_inverted_placement')
        self.assertEqual(episode_capability(self.ep,self.state,'flip')['ensures'],['inverted_held'])
    def test_episode_snapshots_ignore_current_contract_edits(self):
        original=episode_capability(self.ep,self.state,'observe')
        changed={**original,'requires':['future_library_requirement']}
        with patch('skill_harness.runtime.resolve_capability',return_value=changed):
            self.assertEqual(episode_capability(self.ep,self.state,'observe'),original)
        snapshot=read(self.ep/'catalog_snapshot.json')
        self.assertIn('plans/pick_invert_place.json',snapshot['source_sha256'])
        self.assertIn('plans/empty_can_inversion.json',snapshot['source_sha256'])
        self.assertIn('aliases.json',snapshot['source_sha256'])
        self.assertEqual(snapshot['aliases']['flip'],'invert_held_object')
    def test_snapshot_run_and_assessment_survive_removed_live_catalog(self):
        # All current definitions and aliases can disappear after episode creation.
        # Exercise dispatch, assessment, alias resolution, and task completion.
        from skill_harness.tests.test_runtime import Fake
        adapter=Fake(self.root)
        with patch('skill_harness.runtime.resolve_capability',side_effect=AssertionError('live catalog accessed')), \
             patch('skill_harness.runtime.load_catalog',side_effect=AssertionError('live catalog accessed')), \
             patch('skill_harness.catalog.load_catalog',return_value={'capabilities':{},'aliases':{'flip':'unrelated'}}):
            for ident in ('observe','approach','grasp','lift_test','flip','place','release','retreat','verify'):
                state=read(self.ep/'state.json');skill=episode_capability(self.ep,state,ident)
                ref=state.get('last_result');facts={}
                for name in skill['requires']:facts.update(self.facts(name,ref,state['revision']))
                payload={'action':skill['action']}
                if skill['action']=='move':payload.update(pose=[0]*6,grip_required=skill['hold_required'])
                if skill['action']=='grip':payload.update(pos=0 if ident=='release' else 100,grip_required=False)
                state=run_skill(self.ep,ident,payload,facts,ref,adapter,True)
                post={}
                for name in skill['ensures']:post.update(self.facts(name,state['last_result'],state['revision']))
                assess(self.ep,post)
        self.assertEqual(read(self.ep/'state.json')['phase'],'completed')
        self.assertEqual(read(self.ep/'outcome.json')['plan_id'],'empty_can_inversion')
        self.assertEqual(len(adapter.calls),9)
    def test_snapshot_without_aliases_keeps_fixed_legacy_names(self):
        snapshot=read(self.ep/'catalog_snapshot.json');snapshot.pop('aliases')
        write(self.ep/'catalog_snapshot.json',snapshot)
        with patch('skill_harness.runtime.resolve_capability',side_effect=AssertionError('live catalog accessed')):
            self.assertEqual(episode_capability(self.ep,self.state,'flip')['id'],'invert_held_object')
            self.assertEqual(episode_capability(self.ep,self.state,'verify')['id'],'verify_inverted_placement')
            with self.assertRaises(ValueError):episode_capability(self.ep,self.state,'new_live_alias')
    def test_pre_layer_episode_uses_archived_contract(self):
        state={'phase':'ready'}
        with patch('skill_harness.runtime.resolve_capability',return_value={'id':'invert_held_object','ensures':['future_change']}):
            self.assertEqual(episode_capability(self.ep,state,'flip')['ensures'],['inverted_held'])
    def test_scene_and_device_are_not_seeded_from_success(self):
        scene=read(self.ep/'scene.json');self.assertEqual(scene['status'],'unobserved')
        self.assertIsNone(scene['object_pose']);self.assertIsNone(scene['table'])
        self.assertFalse((self.ep/'device_profile.json').exists())
        profile=Path(__file__).resolve().parents[1]/'devices/profile.template.json'
        data=read(profile);self.assertEqual(data['status'],'unverified');self.assertIsNone(data['camera_to_tcp'])

if __name__=='__main__':unittest.main()
