import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from flip_pipeline.evaluation import NumericReportError, validate_numeric_report
from flip_pipeline.modeling import build_rigid_model
from flip_pipeline.physics import assess_properties, combine_weighed_components, estimate_static_payload
from flip_pipeline.storage import evidence, write


class NumericEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        write(self.root/'sensor.json',{'reading':'synthetic unit test fixture'})

    def tearDown(self):
        self.tmp.cleanup()

    def model_fixture(self):
        angles,z=np.meshgrid(np.linspace(0,2*np.pi,64,endpoint=False),np.linspace(.02,.16,20))
        p=np.column_stack([.4+.03*np.cos(angles.ravel()),.02+.03*np.sin(angles.ravel()),z.ravel()])
        observations=[]
        for i in range(2):
            T=np.eye(4);T[0,3]=i*.1
            observations.append({'observation_id':f'v{i}','world_frame':'base','units':'m','target_segmented':True,
                'points_m':(p-T[:3,3]).tolist(),'T_world_camera':T.tolist(),'evidence':['sensor.json']})
        request={'observations':observations,'frame':'base','primitive':'cylinder','axis_hint':[0,0,1],
                 'endpoint_observations':{'both_observed':True,'frame':'base',
                     'points_m':[[.4,.02,.02],[.4,.02,.16]],'evidence':['sensor.json']}}
        result=build_rigid_model(**request)
        write(self.root/'model_request.json',request);write(self.root/'model_result.json',result)
        data={'model_request_path':'model_request.json','model_result_path':'model_result.json'}
        refs=['sensor.json','model_request.json','model_result.json']
        return data,refs,request,result

    def property_fixture(self):
        observations=[{'field':key,'value':False,'source':'human_attestation','evidence':['sensor.json']}
                      for key in ('liquid_present','fragile','hazardous_contents','contents_mobile')]
        observations.append({'field':'material','value':'aluminium shell','source':'measurement','evidence':['sensor.json']})
        request={'components':[{'status':'measured','frame':'object','mass_kg':.02,'com_m':[0,0,.05],
                                'evidence':['sensor.json']}],'frame':'object'}
        write(self.root/'properties.json',observations)
        write(self.root/'property_result.json',assess_properties(observations))
        write(self.root/'mass_request.json',request)
        write(self.root/'mass_result.json',combine_weighed_components(**request))
        data={'property_observations_path':'properties.json','property_result_path':'property_result.json',
              'mass_method':'weighed-components','mass_request_path':'mass_request.json',
              'mass_result_path':'mass_result.json','mass_status':'measured'}
        refs=['sensor.json','properties.json','property_result.json','mass_request.json','mass_result.json']
        return data,refs,observations,request

    def test_actual_model_computation_accepts(self):
        data,refs,_,_=self.model_fixture()
        result=validate_numeric_report(self.root,'model',data,evidence(self.root,refs))
        self.assertTrue(result['valid'])
        self.assertEqual(result['model_kind'],'cylinder')

    def test_handwritten_pass_without_numeric_files_rejected(self):
        with self.assertRaisesRegex(NumericReportError,'model_request_path'):
            validate_numeric_report(self.root,'model',{'checks':{'multi_view_fit':{'result':'pass'}}},['sensor.json'])

    def test_tampered_model_result_rejected_even_when_rehashed(self):
        data,refs,_,result=self.model_fixture()
        result['geometry']['radius_m']=.3
        write(self.root/'model_result.json',result)
        with self.assertRaisesRegex(NumericReportError,'radius_m'):
            validate_numeric_report(self.root,'model',data,refs)

    def test_unlisted_nested_evidence_rejected(self):
        data,refs,_,_=self.model_fixture()
        with self.assertRaisesRegex(NumericReportError,'unhashed nested evidence'):
            validate_numeric_report(self.root,'model',data,refs[1:])

    def test_stale_evidence_hash_rejected(self):
        data,refs,_,_=self.model_fixture()
        hashed=evidence(self.root,refs)
        write(self.root/'sensor.json',{'changed':True})
        with self.assertRaisesRegex(ValueError,'Evidence changed'):
            validate_numeric_report(self.root,'model',data,hashed)

    def test_partial_geometry_cannot_pass_by_matching_recomputation(self):
        data,refs,request,_=self.model_fixture()
        request['observations']=request['observations'][:1]
        write(self.root/'model_request.json',request)
        write(self.root/'model_result.json',build_rigid_model(**request))
        with self.assertRaisesRegex(NumericReportError,'coverage'):
            validate_numeric_report(self.root,'model',data,refs)

    def test_properties_and_measured_component_mass_accept(self):
        data,refs,_,_=self.property_fixture()
        out=validate_numeric_report(self.root,'properties',data,refs)
        self.assertTrue(out['valid'])
        self.assertAlmostEqual(out['mass_kg'],.02)

    def test_visual_prior_cannot_clear_unknown_contents(self):
        data,refs,observations,_=self.property_fixture()
        observations[0]['source']='visual_prior'
        write(self.root/'properties.json',observations)
        write(self.root/'property_result.json',assess_properties(observations))
        with self.assertRaisesRegex(NumericReportError,'unresolved'):
            validate_numeric_report(self.root,'properties',data,refs)

    def test_unknown_material_not_disguised_as_pass(self):
        data,refs,observations,_=self.property_fixture()
        observations[-1]['source']='visual_prior'
        write(self.root/'properties.json',observations)
        write(self.root/'property_result.json',assess_properties(observations))
        with self.assertRaisesRegex(NumericReportError,'material screen'):
            validate_numeric_report(self.root,'properties',data,refs)

    def test_bounded_assumption_not_supported_as_measured(self):
        data,refs,_,_=self.property_fixture()
        data['mass_status']='bounded_hypothesis'
        with self.assertRaisesRegex(NumericReportError,'bounded'):
            validate_numeric_report(self.root,'properties',data,refs)

    def test_tampered_center_of_mass_rejected(self):
        data,refs,_,request=self.property_fixture()
        result=combine_weighed_components(**request);result['com_m']=[0,0,0]
        write(self.root/'mass_result.json',result)
        with self.assertRaisesRegex(NumericReportError,'com_m'):
            validate_numeric_report(self.root,'properties',data,refs)

    def test_nonfinite_or_boolean_numeric_substitution_rejected(self):
        data,refs,_,result=self.model_fixture()
        result['geometry']['radius_m']=True
        write(self.root/'model_result.json',result)
        with self.assertRaises(NumericReportError):
            validate_numeric_report(self.root,'model',data,refs)
        result['geometry']['radius_m']=float('nan')
        (self.root/'model_result.json').write_text(json.dumps(result),encoding='utf8')
        with self.assertRaises(NumericReportError):
            validate_numeric_report(self.root,'model',data,refs)

    def test_entire_tool_payload_not_mislabelled_object(self):
        data,refs,_,_=self.property_fixture()
        cal={'status':'verified','source':'calibrated_ft_sensor','frame':'sensor',
             'force_units':'N','torque_units':'N*m','gravity_compensated':False,
             'target':'complete_payload','mass_distribution_fixed':True,'evidence':['sensor.json']}
        def sample(g,i):
            g=np.array(g,float);g=g/np.linalg.norm(g)*9.81
            return {'sample_id':i,'frame':'sensor','static':True,'no_external_contact':True,
                    'gravity_m_s2':g.tolist(),'force_N':(.4*g).tolist(),
                    'torque_Nm':np.cross([0,0,.03],.4*g).tolist(),'evidence':['sensor.json']}
        request={'samples':[sample(g,str(i)) for i,g in enumerate(([1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]))],
                 'sensor_calibration':cal,'holdout_samples':[sample([1,2,3],'h1'),sample([2,1,3],'h2')]}
        write(self.root/'mass_request.json',request);write(self.root/'mass_result.json',estimate_static_payload(**request))
        data['mass_method']='static-payload'
        with self.assertRaisesRegex(NumericReportError,"tool payload"):
            validate_numeric_report(self.root,'properties',data,refs)


if __name__=='__main__':
    unittest.main()
