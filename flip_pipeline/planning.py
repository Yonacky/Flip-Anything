"""Measured geometry calculations. Outputs are proposals, not controller-path certificates."""
import math
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .calibration import validate_transform

def _points(value):
    p=np.asarray(value,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or len(p)<4 or not np.isfinite(p).all():raise ValueError("Expected finite Nx3 envelope vertices")
    return p

def transform_points(T,points):
    T=validate_transform(T);p=_points(points);return p@T[:3,:3].T+T[:3,3]

def box_vertices(center,half_extents):
    c=np.asarray(center,float);h=np.asarray(half_extents,float)
    if c.shape!=(3,) or h.shape!=(3,) or not np.isfinite(c).all() or not np.isfinite(h).all() or np.any(h<=0):raise ValueError("Invalid box")
    return c+np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*h

def _plane(plane):
    p=np.asarray(plane,float)
    if p.shape!=(4,) or not np.isfinite(p).all() or abs(np.linalg.norm(p[:3])-1)>1e-6:raise ValueError("Plane must be unit [nx,ny,nz,d], n pointing above table")
    return p

def support_margin(com_xy,polygon_xy,uncertainty_m=0):
    from scipy.spatial import ConvexHull
    p=np.asarray(polygon_xy,float);c=np.asarray(com_xy,float)
    if p.ndim!=2 or p.shape[1]!=2 or c.shape!=(2,) or not np.isfinite(p).all() or not np.isfinite(c).all() or uncertainty_m<0:raise ValueError("Invalid support polygon")
    hull=ConvexHull(p)
    distances=-(hull.equations[:,:2]@c+hull.equations[:,2])/np.linalg.norm(hull.equations[:,:2],axis=1)
    return {"minimum_margin_m":float(distances.min()-uncertainty_m),"stable_under_static_assumptions":bool(distances.min()>uncertainty_m),"scope":"gravity projection only, no liquid/slip/dynamic claim"}

def propose_rigid_flip(base_from_object,tool_from_object,object_vertices,tool_vertices,table_plane,clearance_m,axis_base):
    B=validate_transform(base_from_object);G=validate_transform(tool_from_object)
    obj=_points(object_vertices);tool=_points(tool_vertices);plane=_plane(table_plane)
    n=plane[:3];axis=np.asarray(axis_base,float)
    if axis.shape!=(3,) or not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-8:raise ValueError("Invalid flip axis")
    axis=axis/np.linalg.norm(axis)
    if abs(axis@n)>1e-6 or not math.isfinite(clearance_m) or clearance_m<=0:raise ValueError("Flip axis must lie in support plane; positive measured margin required")
    Rflip=Rotation.from_rotvec(axis*math.pi).as_matrix()
    goal=B.copy();goal[:3,:3]=Rflip@B[:3,:3]
    bottom=float(np.min(transform_points(goal,obj)@n+plane[3]))
    goal[:3,3]-=n*bottom
    goal_tool=goal@np.linalg.inv(G)
    camera_clearance=float(np.min(transform_points(goal_tool,tool)@n+plane[3]))
    if camera_clearance<clearance_m:raise ValueError("Chosen grasp cannot place inverted object: tool/camera below required clearance; change grasp")
    # Sphere around object origin bounds all held geometry over every rotation.
    obj_tool=transform_points(np.linalg.inv(G),tool)
    radius=max(np.linalg.norm(obj,axis=1).max(),np.linalg.norm(obj_tool,axis=1).max())
    center_height=float(B[:3,3]@n+plane[3]);lift=max(0.,float(radius+clearance_m-center_height))
    lifted=B.copy();lifted[:3,3]+=n*lift
    turned=lifted.copy();turned[:3,:3]=goal[:3,:3]
    poses=[B,lifted,turned,goal]
    return {"status":"proposal","units":"m_rad","base_from_object":[p.tolist() for p in poses],
            "base_from_tool":[(p@np.linalg.inv(G)).tolist() for p in poses],"required_lift_m":lift,
            "placement_tool_clearance_m":camera_clearance,"bounded_radius_m":float(radius),
            "validated_scope":"table plane versus rigid object/tool only",
            "requires":["current grasp transform","actual robot joint path and obstacles","support/contact observation before release"]}

def sampled_tool_clearance(poses,vertices,table_plane,uncertainty_m=0):
    plane=_plane(table_plane);verts=_points(vertices)
    if uncertainty_m<0:raise ValueError("Negative uncertainty")
    heights=[float(np.min(transform_points(T,verts)@plane[:3]+plane[3])-uncertainty_m) for T in poses]
    if not heights:raise ValueError("No poses")
    return {"minimum_clearance_m":min(heights),"per_sample_m":heights,"scope":"samples only; interpolation unverified"}
