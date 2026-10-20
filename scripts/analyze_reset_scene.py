"""Offline current-scene fit, using explicitly provisional fixed hand-eye estimate."""
import argparse,json
from pathlib import Path
import numpy as np,cv2
from flip_anything.geometry import pose_matrix
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--index',type=int,default=1);a=p.parse_args()
r=Path(a.run);events=[json.loads(x) for x in (r/'events.jsonl').read_text().splitlines()]
meta=next(e['payload']['metadata'] for e in events if e['kind']=='camera_started')['factory_camera_param']
pose=[e['payload'] for e in events if e['kind']=='robot_sample'][-1];pose=pose.get('pose_base_mm_deg',pose.get('pose'))
old=json.loads(Path('calibration/geometry_estimate.json').read_text());X=np.array(old['T_tcp_camera']);W=np.array(pose_matrix(pose))@X
intr=meta['color_intrinsic'];K=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1.]])
dist=np.array([meta['color_distortion'][k] for k in ('k1','k2','p1','p2','k3')])
im=cv2.imread(str(r/f'frame_{a.index:06d}_color.png'));depth=np.load(r/f'frame_{a.index:06d}_depth_raw.npy')
def lift(uv):
 uv=np.array(uv);z=depth[uv[:,1],uv[:,0]].astype(float);q=cv2.undistortPoints(uv.astype(float).reshape(-1,1,2),K,dist).reshape(-1,2)
 pts=np.column_stack((q[:,0]*z,q[:,1]*z,z));valid=(z>100)&(z<1200)
 return pts[valid]@W[:3,:3].T+W[:3,3]
y,x=np.mgrid[550:710:3,50:1240:3];uv=np.column_stack((x.ravel(),y.ravel()));uv=uv[(uv[:,0]<500)|(uv[:,0]>960)]
pts=lift(uv);rng=np.random.default_rng(14);best=None
for _ in range(1000):
 aa,bb,cc=pts[rng.choice(len(pts),3,replace=False)];n=np.cross(bb-aa,cc-aa);n/=np.linalg.norm(n);d=-n@aa
 if abs(n[2])<.95:continue
 ids=np.abs(pts@n+d)<2.5
 if best is None or ids.sum()>best.sum():best=ids
if best is None:raise RuntimeError('No table plane')
t=pts[best];center=t.mean(0);n=np.linalg.svd(t-center)[2][-1]
if n[2]<0:n=-n
d=-n@center
hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);mask=cv2.inRange(hsv,np.array([35,65,35]),np.array([100,255,255]));roi=np.zeros_like(mask);roi[250:710,530:790]=255;mask &=roi
ys,xs=np.where(mask>0);obj=lift(np.column_stack((xs[::3],ys[::3])));height=obj@n+d;obj=obj[(height>3)&(height<350)&(obj[:,0]>400)&(obj[:,0]<850)]
u=np.array([1.,0,0]);u-=n*(u@n);u/=np.linalg.norm(u);v=np.cross(n,u);xy=np.column_stack((obj@u,obj@v));best=None
for _ in range(1500):
 sample=xy[rng.choice(len(xy),3,replace=False)];A=np.column_stack((2*sample,np.ones(3)))
 try:cx,cy,c=np.linalg.solve(A,np.sum(sample**2,axis=1))
 except np.linalg.LinAlgError:continue
 rr=c+cx*cx+cy*cy
 if not 10**2<rr<70**2:continue
 ids=abs(np.linalg.norm(xy-[cx,cy],axis=1)-np.sqrt(rr))<2
 if best is None or ids.sum()>best.sum():best=ids
if best is None:raise RuntimeError('No cylinder')
z=xy[best];cx,cy,c=np.linalg.lstsq(np.column_stack((2*z,np.ones(len(z)))),np.sum(z*z,axis=1),rcond=None)[0];radius=float(np.sqrt(c+cx*cx+cy*cy));bottom=cx*u+cy*v-d*n
heights=obj[best]@n+d
result={'source_run':str(r.resolve()),'pose':pose,'T_base_camera':W.tolist(),'hand_eye_status':'prior fixed-mount estimate; cross-view validation required','table':{'normal':n.tolist(),'d_mm':float(d),'residual_mm':float(np.median(abs(t@n+d))),'support':len(t)},'can':{'bottom_center':bottom.tolist(),'axis':n.tolist(),'radius_mm':radius,'height_green_percentiles_mm':np.percentile(heights,[1,50,99]).tolist(),'circle_residual_mm':float(np.median(abs(np.linalg.norm(z-[cx,cy],axis=1)-radius))),'support':int(best.sum())}}
(r/'scene_fit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
