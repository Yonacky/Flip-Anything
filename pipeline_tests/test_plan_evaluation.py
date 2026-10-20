"""Synthetic plan/report tests only; these do not certify a real robot path."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from flip_pipeline.controller_boundary import matrix_to_controller_move
from flip_pipeline.plan_evaluation import validate_plan_report
from flip_pipeline.planning import box_vertices, propose_rigid_flip
from flip_pipeline.storage import digest, evidence, read, write


class PlanEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.state={'task_id':'TASK_fixture','object_id':'OBJECT_fixture','installation_id':'INSTALL_fixture'}
        write(self.root/'devices/pipeline_policy.json',{'authorized_path_reviewers':['teacher_fixture']})
        write(self.root/'observations.json',{'synthetic':True,'purpose':'unit test geometry; no physical certification'})
        write(self.root/'path_inspection.json',{'synthetic':True,'purpose':'test review binding only'})
        B=np.eye(4);B[2,3]=.1
        self.request={'base_from_object':B.tolist(),'tool_from_object':np.eye(4).tolist(),
            'object_vertices':box_vertices([0,0,0],[.03,.03,.1]).tolist(),
            'tool_vertices':box_vertices([0,0,0],[.02,.02,.02]).tolist(),
            'table_plane':[0,0,1,0],'clearance_m':.01,'axis_base':[1,0,0]}
        self.result=propose_rigid_flip(**self.request)
        write(self.root/'geometry_request.json',self.request)
        write(self.root/'geometry_result.json',self.result)
        segments=[]
        for i,T in enumerate(self.result['base_from_tool'][1:]):
            segments.append({'id':f'segment_{i}','role':['lift','invert','place'][i],
                'request':matrix_to_controller_move(T,grip_required=True,
                    motion_space='joint' if i==1 else 'cartesian_translation',speed=5)})
        self.data={'geometry_request_path':'geometry_request.json',
                   'geometry_result_path':'geometry_result.json',
                   'path_review_path':'path_review.json','segments':segments}
        self.review={**self.state,'reviewer_role':'teacher','reviewer_id':'teacher_fixture',
            'segments_sha256':digest(segments),'geometry_sha256':digest(self.result),
            'result':'accepted','method':'teacher_review',
            'checks':{key:{'result':'pass','evidence':['path_inspection.json']}
                      for key in ('robot_links','tool_and_camera','held_object','obstacles','actual_interpolation')}}
        write(self.root/'path_review.json',self.review)
        self.paths=['observations.json','path_inspection.json','geometry_request.json',
                    'geometry_result.json','path_review.json']

    def tearDown(self):
        self.temp.cleanup()

    def validate(self,data=None,refs=None,state=None):
        return validate_plan_report(self.root,data or self.data,
            refs if refs is not None else evidence(self.root,self.paths),state or self.state)

    def test_recomputed_geometry_with_bound_teacher_review_accepts(self):
        result=self.validate()
        self.assertTrue(result['geometry_recomputed'])
        self.assertEqual(result['full_robot_backend'],'teacher_review')
        self.assertFalse(result['automatic_full_robot_certification'])

    def test_tampered_numeric_result_rejected_even_with_fresh_file_hash(self):
        changed=copy.deepcopy(self.result)
        changed['placement_tool_clearance_m']+=.05
        write(self.root/'geometry_result.json',changed)
        with self.assertRaisesRegex(ValueError,'differs from recomputation'):
            self.validate()

    def test_tampered_plan_request_invalidates_saved_geometry(self):
        changed=copy.deepcopy(self.request);changed['clearance_m']=.02
        write(self.root/'geometry_request.json',changed)
        with self.assertRaisesRegex(ValueError,'differs from recomputation'):
            self.validate()

    def test_stale_review_geometry_digest_rejected(self):
        self.review['geometry_sha256']='0'*64
        write(self.root/'path_review.json',self.review)
        with self.assertRaisesRegex(ValueError,'not bound'):
            self.validate()

    def test_segment_modification_invalidates_review(self):
        data=copy.deepcopy(self.data)
        data['segments'][0]['request']['pose'][0]+=1
        with self.assertRaisesRegex(ValueError,'not bound'):
            self.validate(data=data)

    def test_all_review_identity_fields_are_bound(self):
        for key in ('task_id','object_id','installation_id'):
            with self.subTest(field=key):
                review=copy.deepcopy(self.review);review[key]='wrong'
                write(self.root/'path_review.json',review)
                with self.assertRaisesRegex(ValueError,'identity mismatch'):
                    self.validate()

    def test_absent_review_or_unhashed_review_rejected(self):
        data=copy.deepcopy(self.data);data.pop('path_review_path')
        with self.assertRaisesRegex(ValueError,'path_review_path'):
            self.validate(data=data)
        refs=evidence(self.root,[x for x in self.paths if x!='path_review.json'])
        with self.assertRaisesRegex(ValueError,'path_review_path'):
            self.validate(refs=refs)

    def test_interpolation_must_be_reviewed_with_existing_hashed_evidence(self):
        for failure in ('missing','failed','unhashed_evidence'):
            with self.subTest(failure=failure):
                review=copy.deepcopy(self.review)
                if failure=='missing':review['checks'].pop('actual_interpolation')
                elif failure=='failed':review['checks']['actual_interpolation']['result']='fail'
                else:
                    write(self.root/'unlisted_interpolation.json',{'fixture':True})
                    review['checks']['actual_interpolation']['evidence']=['unlisted_interpolation.json']
                write(self.root/'path_review.json',review)
                with self.assertRaisesRegex(ValueError,'actual_interpolation'):
                    self.validate()

    def test_unapproved_reviewer_or_claimed_automatic_solver_rejected(self):
        cases=[{'reviewer_id':'unapproved'}, {'reviewer_role':'student'},
               {'method':'automatic_solver'}, {'result':'pending'}]
        for changes in cases:
            with self.subTest(changes=changes):
                write(self.root/'path_review.json',{**self.review,**changes})
                with self.assertRaises(ValueError):self.validate()

    def test_controller_pose_and_speed_units_rejected_before_dispatch(self):
        cases=[{'units':'m_rad'}, {'frame':'camera'},
               {'orientation_convention':'intrinsic_xyz'}, {'speed_units':'deg_s'}]
        for changes in cases:
            with self.subTest(changes=changes):
                data=copy.deepcopy(self.data)
                data['segments'][0]['request'].update(changes)
                # Rebind review so this exercises unit validation rather than stale review hash.
                review={**self.review,'segments_sha256':digest(data['segments'])}
                write(self.root/'path_review.json',review)
                with self.assertRaises(ValueError):self.validate(data=data)

    def test_changed_hashed_review_or_measurement_is_detected(self):
        for filename in ('path_review.json','path_inspection.json'):
            with self.subTest(file=filename):
                write(self.root/'path_review.json',self.review)
                write(self.root/'path_inspection.json',{'synthetic':True})
                refs=evidence(self.root,self.paths)
                if filename=='path_review.json':
                    changed=copy.deepcopy(self.review);changed['note']='changed after hashing'
                else:changed={'synthetic':True,'changed':True}
                write(self.root/filename,changed)
                with self.assertRaisesRegex(ValueError,'Evidence changed'):
                    self.validate(refs=refs)

    def test_camera_envelope_collision_blocks_geometry_before_review(self):
        changed=copy.deepcopy(self.request)
        changed['tool_vertices']=box_vertices([0,0,.15],[.02,.02,.05]).tolist()
        write(self.root/'geometry_request.json',changed)
        with self.assertRaisesRegex(ValueError,'tool/camera'):
            self.validate()


if __name__=='__main__':
    unittest.main()
