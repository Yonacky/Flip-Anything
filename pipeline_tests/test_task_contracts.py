"""Synthetic task-contract tests. No physical object or hardware authorization."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from flip_pipeline import workflow as w
from flip_pipeline.contracts import (PROFILE, PROFILE_PATH, CONDITIONS, load_contract,
    validate_scope, validate_plan_contract, brief)
from flip_pipeline.storage import read, write, evidence
from test_calibration import approve, profile_fixture
import test_evaluation

class TaskContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.repo=Path(__file__).resolve().parents[1]
        write(self.root/PROFILE_PATH,read(self.repo/PROFILE_PATH))
        write(self.root/'devices/pipeline_policy.json',{'authorized_scope_reviewers':['human']})
        write(self.root/'devices/installation.json',{'status':'unverified'})
        write(self.root/'operator.json',{'synthetic':True,'statement':'unit test only'})
        write(self.root/'observation.json',{'synthetic':True,'object_id':'object-1'})
        self.task=w.new_task(self.root,'object-1','synthetic-installation',profile=PROFILE)
        self.state=read(self.task/'state.json')
        self.scope={'profile':PROFILE,'source':'operator_declaration','approved_by':'human',
            'installation_id':'synthetic-installation','object_ids':['object-1'],
            'operator_supervised':True,'conditions':{k:True for k in CONDITIONS},'evidence':['operator.json']}
        write(self.root/'scope.json',self.scope)
        self.refs=['operator.json','observation.json','scope.json']
        self.report={'source':'physical_evidence','checks':{'scope_membership':{
            'result':'pass','method':'operator_scope_and_observation','evidence':self.refs,
            'detail':'Synthetic declaration and observation fixture.'}},
            'scope_declaration_path':'scope.json','scope_contradictions':[],
            'observation_evidence':['observation.json'],'mass_status':'not_required','material_status':'not_required'}

    def tearDown(self): self.tmp.cleanup()

    def validate(self):return validate_scope(self.root,self.report,evidence(self.root,self.refs),self.state)

    def prepare_geometry(self):
        profile,installation=profile_fixture();approve(profile)
        write(self.root/'profile.json',profile);write(self.root/'installation.json',installation)
        refs=['profile.json','installation.json','observation.json']
        checks={k:{'result':'pass','method':sorted(w.ALLOWED_METHODS[k])[0],
            'evidence':refs,'detail':'Synthetic calibration fixture'} for k in w.REQUIRED['calibration']}
        w.attach_report(self.root,self.task,'calibration',{'source':'physical_evidence','checks':checks,
            'profile_path':'profile.json','installation_path':'installation.json'},refs)
        write(self.root/"sensor.json",{"synthetic":True})
        data,refs,_,_=test_evaluation.NumericEvaluationTests.model_fixture(self)
        data.update(source='physical_evidence',checks={k:{'result':'pass','method':sorted(w.ALLOWED_METHODS[k])[0],
            'evidence':refs,'detail':'Synthetic model fixture'} for k in w.REQUIRED['model']})
        w.attach_report(self.root,self.task,'model',data,refs)

    def test_geometric_strategy_accepts_without_mass_or_material_measurement(self):
        result=self.validate()
        self.assertEqual(result['mass_status'],'not_required')
        self.assertNotIn('com_m',result)

    def test_new_task_pins_contract_and_exposes_only_relevant_property_checks(self):
        self.prepare_geometry()
        self.assertEqual(w.status(self.root,self.task)['required_checks'],['scope_membership'])
        w.attach_report(self.root,self.task,'properties',self.report,self.refs)
        self.assertEqual(w.status(self.root,self.task)['stage'],'plan')
        self.assertEqual(brief(self.root,self.task)['property_policy'],'not_required_by_geometry_strategy')
        self.assertEqual(brief(self.root,self.task)['blockers'][0]['code'],'path_review_required')

    def test_old_mode_still_requires_measurements(self):
        task=w.new_task(self.root,'object-2','synthetic-installation')
        state=read(task/'state.json')
        with self.assertRaises(ValueError):w.validate_report('properties',self.report,evidence(self.root,self.refs),state)
        self.assertEqual(brief(self.root,task)['property_policy'],'measured_properties')

    def test_empty_evidence_not_accepted_as_scope(self):
        self.scope['evidence']=[];write(self.root/'scope.json',self.scope)
        with self.assertRaisesRegex(ValueError,'operator evidence'):self.validate()

    def test_self_referencing_declaration_is_not_operator_evidence(self):
        self.scope['evidence']=['scope.json'];write(self.root/'scope.json',self.scope)
        with self.assertRaises(ValueError):self.validate()

    def test_new_object_needs_explicit_batch_membership(self):
        self.scope['object_ids']=['other-object'];write(self.root/'scope.json',self.scope)
        with self.assertRaisesRegex(ValueError,'cover'):self.validate()

    def test_batch_scope_can_be_reused_for_explicit_objects(self):
        self.scope['object_ids']=['object-1','object-2'];write(self.root/'scope.json',self.scope)
        self.assertTrue(self.validate()['valid'])

    def test_wrong_installation_or_student_approval_rejected(self):
        for key,value in [('installation_id','other'),('approved_by','DS'),('source','visual_prior'),('operator_supervised',False)]:
            with self.subTest(key=key):
                changed={**self.scope,key:value};write(self.root/'scope.json',changed)
                with self.assertRaises(ValueError):self.validate()

    def test_observed_contradiction_blocks(self):
        self.report['scope_contradictions']=['liquid observed']
        with self.assertRaisesRegex(ValueError,'contradiction'):self.validate()

    def test_unchecked_observation_not_automatically_empty(self):
        self.report.pop('scope_contradictions')
        with self.assertRaises(ValueError):self.validate()

    def test_each_scope_condition_required(self):
        for key in CONDITIONS:
            with self.subTest(key=key):
                changed=copy.deepcopy(self.scope);changed['conditions'][key]=None
                write(self.root/'scope.json',changed)
                with self.assertRaisesRegex(ValueError,'incomplete'):self.validate()

    def test_current_observation_required(self):
        self.report['observation_evidence']=[]
        with self.assertRaisesRegex(ValueError,'observation'):self.validate()

    def test_fabricated_mass_not_accepted(self):
        for key,value in [('mass_kg',.1),('com_m',[0,0,0]),('material','plastic'),('mass_result_path','fake.json')]:
            with self.subTest(key=key):
                data={**self.report,key:value}
                with self.assertRaises(ValueError):validate_scope(self.root,data,evidence(self.root,self.refs),self.state)

    def test_rehashed_changed_scope_cannot_silently_affect_accepted_task(self):
        self.prepare_geometry();w.attach_report(self.root,self.task,'properties',self.report,self.refs)
        self.scope['object_ids']=[];write(self.root/'scope.json',self.scope)
        self.assertIsNotNone(w.status(self.root,self.task)['integrity_error'])

    def test_contract_mutation_detected(self):
        contract=load_contract(self.root,PROFILE);contract['scope']='arbitrary objects'
        write(self.root/PROFILE_PATH,contract)
        self.assertIn('contract changed',w.status(self.root,self.task)['integrity_error'])

    def test_required_geometry_cannot_be_skipped(self):
        with self.assertRaisesRegex(ValueError,'Missing previous stage'):
            w.attach_report(self.root,self.task,'properties',self.report,self.refs)

    def test_plan_cannot_depend_on_omitted_properties(self):
        data={'strategy':'geometry_guided_pick_invert_place','property_requirements':[]}
        validate_plan_contract(data,self.state)
        for changes in ({'property_requirements':['com']},{'strategy':'dynamic_throw'},{'mass_kg':.1},{'simulation_result_path':'fake.json'}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):validate_plan_contract({**data,**changes},self.state)

    def test_brief_reports_installation_block_instead_of_property_research(self):
        out=brief(self.root,self.task)
        self.assertEqual(out['blockers'][0]['code'],'installation_unverified')
        self.assertNotIn('mass_distribution',out['required_checks'])

    def test_closed_task_brief_has_no_new_work(self):
        w.finalize(self.root,self.task,'incomplete',['operator.json'],{'text':'Fixture complete','interventions':[],'failures':[],'unresolved':['calibration']})
        out=brief(self.root,self.task)
        self.assertTrue(out['terminal']);self.assertIsNone(out['question']);self.assertEqual(out['blockers'],[])

    def test_cli_defaults_to_profile_and_templates_follow_task(self):
        code='from flip_pipeline.cli import main;main('+repr(str(self.root))+')'
        def cli(*args):
            result=subprocess.run([sys.executable,'-c',code,*args],cwd=self.repo,capture_output=True,text=True,encoding='utf8',check=True)
            return json.loads(result.stdout)
        out=cli('begin','--object','cli-object','--installation','synthetic-installation')
        task=out['task'];self.assertEqual(read(Path(task)/'state.json')['task_contract']['id'],PROFILE)
        template=cli('report-template','properties','properties.json','--task',task)
        self.assertEqual(set(template['checks']),{'scope_membership'})
        self.assertIsNone(template['scope_contradictions'])
        self.assertNotIn('mass_request_path',template)
        plan=cli('report-template','plan','plan.json','--task',task)
        self.assertEqual(plan['property_requirements'],[])
        strict=cli('begin','--object','research-object','--installation','synthetic-installation','--profile','measured_properties')
        self.assertNotIn('task_contract',read(Path(strict['task'])/'state.json'))

if __name__=='__main__':unittest.main()
