import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from flip_pipeline.modeling import (ModelInputError, backproject_rgbd, build_rigid_model,
    import_mose3_tracks, merge_known_views, register_multiview_open3d, separate_table)
from flip_pipeline.physics import (IdentificationError, assess_properties,
    combine_support_projections, combine_weighed_components, estimate_static_payload,
    estimate_support_projection)


def observation(points, ident="view0", translation=(0,0,0)):
    T = np.eye(4)
    T[:3,3] = translation
    return {"observation_id":ident,"points_m":(np.asarray(points)-T[:3,3]).tolist(),
            "T_world_camera":T.tolist(),"world_frame":"base","units":"m",
            "target_segmented":True,"evidence":[ident+".npz"]}


def cylinder_cloud(angle_range=(0,2*np.pi)):
    theta = np.linspace(*angle_range,144,endpoint=False)
    z = np.linspace(0.02,0.16,35)
    theta,z = np.meshgrid(theta,z)
    return np.column_stack([0.4+0.03*np.cos(theta.ravel()),0.02+0.03*np.sin(theta.ravel()),z.ravel()])


class ModelingTests(unittest.TestCase):
    def test_known_pose_merge(self):
        p = cylinder_cloud()
        result = merge_known_views([observation(p),observation(p,"view1",(.05,0,0))],frame="base")
        merged = np.asarray(result["points_m"])
        self.assertTrue(result["view_diversity"])
        np.testing.assert_allclose(merged.mean(axis=0),[.4,.02,.09],atol=.0004)
        json.dumps(result,allow_nan=False)

    def test_full_cylinder(self):
        p = cylinder_cloud()
        result = build_rigid_model([observation(p),observation(p,"v1",(.1,0,0))],
            frame="base",primitive="cylinder",axis_hint=[0,0,1],
            endpoint_observations={"both_observed":True,"frame":"base","points_m":[[.4,.02,.02],[.4,.02,.16]],"evidence":["endpoints.png"]})
        self.assertEqual(result["status"],"estimated")
        self.assertAlmostEqual(result["geometry"]["radius_m"],.03,places=4)
        self.assertAlmostEqual(result["geometry"]["observed_height_m"],.14,places=3)
        self.assertFalse(result["geometry"]["axial_yaw_observable"])

    def test_endpoint_boolean_alone_does_not_certify_dimensions(self):
        p = cylinder_cloud()
        with self.assertRaises(ModelInputError):
            build_rigid_model([observation(p),observation(p,"v1",(.1,0,0))],
                frame="base",primitive="cylinder",axis_hint=[0,0,1],
                endpoint_observations={"both_observed":True,"evidence":["ends.png"]})

    def test_mismatched_endpoint_axis_rejected(self):
        p = cylinder_cloud()
        with self.assertRaises(ModelInputError):
            build_rigid_model([observation(p),observation(p,"v1",(.1,0,0))],
                frame="base",primitive="cylinder",axis_hint=[0,0,1],
                endpoint_observations={"both_observed":True,"frame":"base",
                    "points_m":[[.5,.02,.02],[.5,.02,.16]],"evidence":["ends.json"]})
    def test_single_partial_view_never_full_model(self):
        result = build_rigid_model([observation(cylinder_cloud((0,np.pi/2)))],frame="base",
                                  primitive="cylinder",axis_hint=[0,0,1])
        self.assertFalse(result["usable_for_planning"])
        self.assertIn("sidewall_coverage_incomplete",result["quality"]["issues"])
        self.assertIn("endpoints_not_independently_observed",result["quality"]["issues"])

    def test_bad_transform_and_unsegmented_rejected(self):
        obs = observation(cylinder_cloud())
        obs["T_world_camera"][0][0] = 2
        with self.assertRaises(ModelInputError):
            merge_known_views([obs],frame="base")
        obs = observation(cylinder_cloud())
        obs["target_segmented"] = False
        with self.assertRaises(ModelInputError):
            merge_known_views([obs],frame="base")

    def test_table_band_is_removed_but_not_endpoint_evidence(self):
        p = cylinder_cloud()
        result = separate_table(p,{"normal":[0,0,1],"offset_m":-.02,"frame":"base",
                    "units":"m","evidence":["table.npz"]},frame="base")
        self.assertGreater(result["removed_count"],0)
        self.assertTrue(np.min(np.asarray(result["points_m"])[:,2])>.023)

    def test_box_fit_with_known_axes(self):
        rng = np.random.default_rng(4)
        clouds = []
        ext = np.array([.08,.05,.11])
        for axis in range(3):
            for sign in (-1,1):
                p = (rng.random((300,3))-.5)*ext
                p[:,axis]=sign*ext[axis]/2
                clouds.append(p)
        p = np.concatenate(clouds)+[.3,.02,.07]
        out = build_rigid_model([observation(p),observation(p,"view1",(.1,0,0))],
                               frame="base",primitive="box",axes_hint=np.eye(3))
        self.assertTrue(out["usable_for_planning"])
        np.testing.assert_allclose(out["geometry"]["extents_m"],ext,atol=.001)

    def test_depth_requires_correct_profile_scale_and_mask(self):
        depth = np.array([[1000,0],[1000,2000]],dtype=np.uint16)
        intr = {"K":[[100,0,0],[0,100,0],[0,0,1]],"profile_id":"depth-2x2",
                "pixel_frame":"camera","width":2,"height":2,"rectified":True}
        out = backproject_rgbd(depth,depth_scale_m=.001,intrinsics=intr,
            valid_mask=np.ones((2,2),bool),min_depth_m=.1,max_depth_m=1.5,
            frame="camera",evidence=["depth.png"])
        np.testing.assert_allclose(out["points_m"],[[0,0,1],[0,.01,1]])
        self.assertEqual(out["valid_fraction_of_mask"],.5)
        bad=copy.deepcopy(intr);bad["rectified"]=False
        with self.assertRaises(ModelInputError):
            backproject_rgbd(depth,depth_scale_m=.001,intrinsics=bad,
                valid_mask=np.ones((2,2),bool),min_depth_m=.1,max_depth_m=1.5,
                frame="camera",evidence=["depth.png"])

    def test_mose3_schema_alignment_occlusion_and_scope(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"predictions.npz"
            tracks = np.ones((2,2,3,3))
            visibility = np.ones((2,2,3));visibility[1,0,1]=-2
            np.savez(path,tracks=tracks,visibility=visibility,se3_valid=np.ones((2,2,3),bool),
                     se3_quat=np.tile([1.,0,0,0],(2,2,3,1)),se3_trans=np.zeros((2,2,3,3)),
                     camera_poses=np.tile(np.eye(4),(2,1,1)),intrinsics=np.eye(3),query_idx=np.int64(0))
            align={"status":"verified","units":"m","world_frame":"base","scale_to_m":.1,
                   "T_world_first_camera":np.eye(4).tolist(),"holdout_rms_m":.001,"evidence":["alignment.json"]}
            out=import_mose3_tracks(path,alignment=align,target_mask=np.ones((2,3),bool),evidence=["video.mp4"])
            self.assertFalse(out["usable_for_dynamics"])
            self.assertIsNone(out["tracks_m"][1][1])
            np.testing.assert_allclose(out["tracks_m"][0][0],[.1,.1,.1])
            json.dumps(out,allow_nan=False)
            align["status"]="assumed"
            with self.assertRaises(ModelInputError):
                import_mose3_tracks(path,alignment=align,target_mask=np.ones((2,3),bool),evidence=["video.mp4"])

    @unittest.skipUnless(importlib.util.find_spec("open3d"),"Open3D optional dependency is unavailable")
    def test_open3d_real_registration(self):
        # Asymmetric smooth patch has local normal constraints in three directions.
        rng=np.random.default_rng(9)
        p=rng.uniform(-1,1,(3000,3))
        p /= np.linalg.norm(p,axis=1)[:,None]
        p *= [.055,.035,.085]
        p += [.3,.02,.12]
        result=register_multiview_open3d([observation(p),observation(p,"v1",(.1,0,0))],frame="base")
        self.assertEqual(result["method"],"open3d_pose_graph_point_plane_icp")
        self.assertLess(result["edges"][0]["rmse_m"],1e-6)


    @unittest.skipUnless(importlib.util.find_spec("open3d"),"Open3D optional dependency is unavailable")
    def test_open3d_corrects_small_pose_error_and_rejects_no_overlap(self):
        rng=np.random.default_rng(11)
        p=rng.uniform(-1,1,(2500,3))
        p/=np.linalg.norm(p,axis=1)[:,None]
        p*= [.055,.035,.085]
        p+= [.3,.02,.12]
        delta=np.array([.001,-.0007,.0004])
        result=register_multiview_open3d([observation(p),observation(p+delta,"v1",(.1,0,0))],frame="base")
        correction=np.array(result["world_corrections"][1])
        corrected=(p+delta)@correction[:3,:3].T+correction[:3,3]
        self.assertLess(np.sqrt(np.mean((corrected-p)**2)),.0003)
        with self.assertRaises(ModelInputError):
            register_multiview_open3d([observation(p),observation(p+[.3,0,0],"v1",(.1,0,0))],frame="base")

CAL={"status":"verified","source":"calibrated_ft_sensor","frame":"sensor",
     "force_units":"N","torque_units":"N*m","gravity_compensated":False,
     "target":"complete_payload","mass_distribution_fixed":True,"evidence":["ft_cal.json"]}
DIRECTIONS=np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],float)


def measurements(directions, prefix="train", mass=.6, com=(.02,-.01,.08), bf=(.1,-.2,.3),bt=(.01,.02,-.015)):
    out=[]
    for i,g in enumerate(directions):
        g=np.asarray(g,float);g=g/np.linalg.norm(g)*9.81
        out.append({"sample_id":prefix+str(i),"frame":"sensor","static":True,"no_external_contact":True,
                    "gravity_m_s2":g.tolist(),"force_N":(mass*g+bf).tolist(),
                    "torque_Nm":(np.cross(com,mass*g)+bt).tolist(),"evidence":[prefix+str(i)+".json"]})
    return out


class PhysicsTests(unittest.TestCase):
    def test_static_mass_com_and_bias_with_independent_holdout(self):
        out=estimate_static_payload(measurements(DIRECTIONS),CAL,
              holdout_samples=measurements([[1,2,3],[-2,3,1]],"hold"))
        self.assertEqual(out["status"],"measured")
        self.assertAlmostEqual(out["mass_kg"],.6,places=8)
        np.testing.assert_allclose(out["com_m"],[.02,-.01,.08],atol=1e-10)
        np.testing.assert_allclose(out["force_bias_N"],[.1,-.2,.3],atol=1e-10)
        np.testing.assert_allclose(out["torque_bias_Nm"],[.01,.02,-.015],atol=1e-10)
        json.dumps(out,allow_nan=False)

    def test_no_holdout_is_provisional(self):
        self.assertEqual(estimate_static_payload(measurements(DIRECTIONS),CAL)["status"],"provisional")

    def test_no_rotation_cannot_solve_com(self):
        with self.assertRaisesRegex(IdentificationError,"underexcited"):
            estimate_static_payload(measurements([[0,0,1]]*6),CAL)

    def test_known_bias_requires_calibration(self):
        cal={**CAL,"bias_verified":True,"force_bias_N":[.1,-.2,.3],"torque_bias_Nm":[.01,.02,-.015]}
        result=estimate_static_payload(measurements(DIRECTIONS),cal,bias_mode="known")
        np.testing.assert_allclose(result["com_m"],[.02,-.01,.08],atol=1e-10)
        with self.assertRaises(IdentificationError):
            estimate_static_payload(measurements(DIRECTIONS),CAL,bias_mode="known")

    def test_nonstatic_distribution_and_reused_validation_rejected(self):
        with self.assertRaises(IdentificationError):
            estimate_static_payload(measurements(DIRECTIONS),{**CAL,"mass_distribution_fixed":False})
        with self.assertRaises(IdentificationError):
            estimate_static_payload(measurements(DIRECTIONS),CAL,holdout_samples=measurements(DIRECTIONS[:2]))
    def test_joint_currents_are_not_wrench(self):
        bad={**CAL,"source":"joint_current_estimate"}
        with self.assertRaises(IdentificationError):
            estimate_static_payload(measurements(DIRECTIONS),bad)

    def test_gravity_compensation_invalidates_input(self):
        with self.assertRaises(IdentificationError):
            estimate_static_payload(measurements(DIRECTIONS),{**CAL,"gravity_compensated":True})

    def test_contact_and_bad_holdout_fail(self):
        bad=measurements(DIRECTIONS);bad[0]["no_external_contact"]=False
        with self.assertRaises(IdentificationError):
            estimate_static_payload(bad,CAL)
        hold=measurements([[1,2,3],[-2,3,1]],"hold");hold[0]["force_N"][0]+=1
        with self.assertRaisesRegex(IdentificationError,"holdout"):
            estimate_static_payload(measurements(DIRECTIONS),CAL,holdout_samples=hold)

    def test_paired_unloaded_removes_tool_mass(self):
        loaded=measurements(DIRECTIONS,mass=1.0,com=(.01,.02,.03))
        unloaded=measurements(DIRECTIONS,prefix="unloaded",mass=.4,com=(0,0,.02))
        for l,u in zip(loaded,unloaded):l["unloaded"]=u
        out=estimate_static_payload(loaded,{**CAL,"target":"object_difference"})
        self.assertAlmostEqual(out["mass_kg"],.6)
        np.testing.assert_allclose(out["com_m"],np.array([.01,.02,.03])/.6-np.array([0,0,.008])/.6,atol=1e-10)

    def test_simple_supports_observe_only_projection(self):
        supports=[{"frame":"object","position_m":[0,0,0],"force_N":[0,0,5.886],"evidence":["scaleL"]},
                  {"frame":"object","position_m":[.1,0,0],"force_N":[0,0,3.924],"evidence":["scaleR"]}]
        out=estimate_support_projection(supports,frame="object",gravity_m_s2=[0,0,-9.81],
                  loadcell_calibration={"status":"verified","external_loads_only_gravity_and_supports":True,"evidence":["scale_cal"]})
        self.assertAlmostEqual(out["mass_kg"],1)
        np.testing.assert_allclose(out["com_projection_m"],[.04,0,0],atol=1e-10)
        self.assertIsNone(out["com_m"])

    def test_two_support_orientations_recover_full_com(self):
        com=np.array([.04,.01,.03])
        projections=[]
        for axis in ([0,0,1],[1,0,0]):
            axis=np.array(axis)
            projections.append({"status":"partial","method":"static_support_weighing","frame":"object",
                "mass_kg":1.,"com_projection_m":((np.eye(3)-np.outer(axis,axis))@com).tolist(),
                "unobserved_axis":axis.tolist(),"evidence":["weighing.json"]})
        out=combine_support_projections(projections,frame="object")
        np.testing.assert_allclose(out["com_m"],com,atol=1e-10)
        with self.assertRaises(IdentificationError):
            combine_support_projections([projections[0],projections[0]],frame="object")

    def test_components_do_not_assume_uniform_density(self):
        parts=[{"status":"measured","frame":"object","mass_kg":1,"com_m":[0,0,0],"evidence":["p1"]},
               {"status":"measured","frame":"object","mass_kg":2,"com_m":[.3,0,0],"evidence":["p2"]}]
        out=combine_weighed_components(parts,frame="object")
        np.testing.assert_allclose(out["com_m"],[.2,0,0])
        parts[1]["status"]="assumed"
        with self.assertRaises(IdentificationError):combine_weighed_components(parts,frame="object")

    def test_visual_prior_cannot_clear_liquid_risk(self):
        out=assess_properties([{"field":"liquid_present","source":"visual_prior","value":False,"evidence":["rgb.png"]}])
        self.assertTrue(out["requires_teacher_or_human"])
        self.assertIn("liquid_present",out["unresolved_hazards"])

    def test_conflicting_material_evidence_and_known_hazard(self):
        claims=[{"field":k,"source":"human_attestation","value":False,"evidence":["attestation.json"]}
                for k in ("liquid_present","fragile","hazardous_contents","contents_mobile")]
        self.assertTrue(assess_properties(claims)["safe_for_exploration"])
        claims.append({"field":"liquid_present","source":"measurement","value":True,"evidence":["contact.json"]})
        result=assess_properties(claims)
        self.assertTrue(result["requires_teacher_or_human"])
        self.assertEqual(result["properties"]["liquid_present"]["status"],"conflict")


if __name__ == "__main__":
    unittest.main()



