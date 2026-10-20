import json,tempfile,unittest
from pathlib import Path
from skill_harness.runtime import new_episode,run_skill,assess,evidence,read,write,digest,retrieve
from skill_harness.learning import propose,promote

class Fake:
    def __init__(self,root,fail=False,unknown=False):self.root=root;self.fail=fail;self.unknown=unknown;self.calls=[]
    def dispatch(self,payload,reference):
        self.calls.append(payload)
        if self.unknown:raise TimeoutError('unknown result')
        result={'ok':not self.fail,'state':{'pose':[0]*6,'gripper':{'gOBJ':3 if payload.get('action')=='grip' and payload.get('pos')==0 else 2,'gFLT':0}},'camera':{'color_file':'fake'}}
        path=self.root/('result_'+str(len(self.calls))+'.json');write(path,result)
        return {'result_file':str(path),'result':result}

class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.folder=self.root/'ep';new_episode(self.folder,'test_run');self.adapter=Fake(self.root)
    def tearDown(self):self.tmp.cleanup()
    def fact(self,name,ref=None,revision=None,source='perception'):
        state=read(self.folder/'state.json');ref=ref or state['last_result']
        return {name:{'value':True,'revision':state['revision'] if revision is None else revision,'result_sha256':digest(ref),'source':source,'method':'fixture checker','artifacts':[str(ref)]}}
    def observe(self):return run_skill(self.folder,'observe',{'action':'observe'},{},None,self.adapter,True)
    def test_candidate_not_executed_by_default(self):
        with self.assertRaises(ValueError):run_skill(self.folder,'observe',{'action':'observe'},{},None,self.adapter)
        self.assertEqual(self.adapter.calls,[])
    def test_no_advance_without_assessment(self):
        self.observe()
        with self.assertRaises(RuntimeError):self.observe()
        self.assertEqual(len(self.adapter.calls),1)
        assess(self.folder,self.fact('scene_visible'))
        self.assertEqual(read(self.folder/'state.json')['phase'],'ready')
    def test_stale_or_wrong_receipt_rejected(self):
        self.observe()
        with self.assertRaises(ValueError):assess(self.folder,self.fact('scene_visible',revision=0))
        f=self.fact('scene_visible');f['scene_visible']['result_sha256']='wrong'
        with self.assertRaises(ValueError):assess(self.folder,f)
    def test_unknown_outcome_cannot_retry(self):
        self.adapter.unknown=True
        with self.assertRaises(TimeoutError):self.observe()
        self.assertEqual(read(self.folder/'state.json')['phase'],'unknown')
        with self.assertRaises(RuntimeError):self.observe()
        self.assertEqual(len(self.adapter.calls),1)
    def test_failed_receipt_blocks(self):
        self.adapter.fail=True;self.observe()
        with self.assertRaises(RuntimeError):assess(self.folder,{})
    def test_release_without_support_never_dispatches(self):
        ref=self.root/'result_seed.json';write(ref,{'ok':True})
        with self.assertRaises(ValueError):run_skill(self.folder,'release',{'action':'grip','pos':0,'grip_required':False},{},ref,self.adapter,True)
        self.assertEqual(self.adapter.calls,[])
    def test_controller_code_not_table_support(self):
        self.observe();s=read(self.folder/'state.json')
        with self.assertRaises(ValueError):evidence(self.fact('supported',source='controller'),['supported'],s,s['last_result'])
    def test_human_evidence_never_silent(self):
        self.observe()
        with self.assertRaises(ValueError):assess(self.folder,self.fact('scene_visible',source='human'))
        assess(self.folder,self.fact('scene_visible',source='human'),True)
        self.assertTrue(read(self.folder/'state.json')['human_interventions'])
    def test_retrieval_excludes_candidates(self):
        self.assertEqual(retrieve([]),[])
        self.assertEqual([x['id'] for x in retrieve([],True)],['observe'])
    def test_lesson_needs_independent_validation(self):
        self.observe();lesson={k:k for k in ('trigger','failure','correction','verification','scope')}
        candidate=self.root/'candidate.json';propose(self.folder,lesson,candidate)
        with self.assertRaises(ValueError):promote(candidate,self.folder,self.root/'approved.json')
        self.assertFalse((self.root/'approved.json').exists())

    def test_complete_mock_episode_cannot_promote_real_world_experience(self):
        from skill_harness.runtime import library
        for sid in ('observe','approach','grasp','lift_test','invert_held_object','place','release','retreat','verify_inverted_placement'):
            skill=library()[sid];state=read(self.folder/'state.json');ref=state.get('last_result')
            facts={}
            for name in skill['requires']:facts.update(self.fact(name))
            payload={'action':skill['action']}
            if skill['action']=='move':payload.update(pose=[0]*6,grip_required=skill['hold_required'])
            if skill['action']=='grip':payload.update(pos=0 if sid=='release' else 100,grip_required=False)
            run_skill(self.folder,sid,payload,facts,ref,self.adapter,True)
            post={}
            for name in skill['ensures']:post.update(self.fact(name))
            assess(self.folder,post)
        self.assertTrue(read(self.folder/'outcome.json')['autonomous_success'])
        source=self.root/'source';new_episode(source,'another_run')
        (source/'events.jsonl').write_text('{}\n')
        lesson={k:k for k in ('trigger','failure','correction','verification','scope')}
        candidate=self.root/'candidate.json';propose(source,lesson,candidate)
        with self.assertRaises(ValueError):
            promote(candidate,self.folder,self.root/'approved.json')
        self.assertEqual(read(self.folder/'outcome.json')['backend'],'mock')

if __name__=='__main__':unittest.main()
