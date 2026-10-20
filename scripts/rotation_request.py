"""Generate one relative rotation request. Never connects or moves hardware."""
import argparse,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from materials import safe,read,write,run_root

def matrix(rpy):
    r,p,y=np.radians(rpy);cr,sr,cp,sp,cy,sy=np.cos(r),np.sin(r),np.cos(p),np.sin(p),np.cos(y),np.sin(y)
    return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                     [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])

def angles(R):
    c=math.hypot(R[0,0],R[1,0]);p=math.atan2(-R[2,0],c)
    if c>1e-9:r,y=math.atan2(R[2,1],R[2,2]),math.atan2(R[1,0],R[0,0])
    else:r,y=0.,math.atan2(-R[0,1],R[1,1])
    return np.degrees([r,p,y]).tolist()

def target(pose,axis,degrees,frame):
    if len(pose)!=6 or not np.isfinite(pose).all():raise ValueError('Need finite six-element pose')
    if axis not in 'xyz' or len(axis)!=1 or frame not in ('tool','base'):raise ValueError('Invalid rotation frame/axis')
    if not math.isfinite(degrees) or abs(degrees)>60:raise ValueError('This helper supports a single step up to 60 degrees; executor limits still apply')
    v=np.eye(3)['xyz'.index(axis)];K=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    a=math.radians(degrees);D=np.eye(3)+math.sin(a)*K+(1-math.cos(a))*(K@K)
    R=matrix(pose[3:]);T=R@D if frame=='tool' else D@R
    result=list(pose[:3])+angles(T)
    if not np.allclose(matrix(result[3:]),T,atol=1e-8):raise ValueError('Rotation conversion verification failed')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',required=True);p.add_argument('--axis',choices=list('xyz'),required=True)
    p.add_argument('--degrees',type=float,required=True);p.add_argument('--frame',choices=['tool','base'],required=True)
    p.add_argument('--held',choices=['yes','no'],required=True);p.add_argument('--output',required=True)
    a=p.parse_args();ref=safe(a.reference);rid=read(ROOT/'ds_control/active.json')['run']
    if ref.parent!=run_root()/rid or not ref.name.startswith('result_'):raise ValueError('Use a receipt from the active run')
    d=read(ref)
    if not d.get('ok') or 'state' not in d:raise ValueError('Need successful state receipt')
    out=safe(a.output)
    if not out.is_relative_to(ROOT/'workspace'):raise ValueError('Save requests under workspace/')
    if out.exists():raise ValueError('Use a new output filename')
    result=target(d['state']['pose'],a.axis,a.degrees,a.frame)
    motion='cartesian_translation' if abs(a.degrees)<.5 else 'joint'
    write(out,{'action':'move','pose':result,'grip_required':a.held=='yes','capture':True,'frame':'base','units':'mm_deg','orientation_convention':'extrinsic_xyz_rpy','motion_space':motion,'speed':10 if motion=='cartesian_translation' else 5,'speed_units':'mm_s' if motion=='cartesian_translation' else 'deg_s'})
    print(json.dumps({'request':str(out),'reference':str(ref),'hardware_moved':False,
                      'note':'Preview only. Check tool/object sweep and reachability, then authorize through pipeline.ps1 before robot.ps1.'}))
if __name__=='__main__':main()
