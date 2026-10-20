"""Local RGB-D/PnP hand-eye and table/cylinder estimates with explicit residuals."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('run')
args=parser.parse_args()
root=Path(args.run)
data=json.loads((root/'geometry_samples.json').read_text())
if not any(s['label']=='heldout' for s in data['samples']):
    if len(data['samples'])<5:raise RuntimeError('Need four training poses and a separate validation pose')
    data['samples'][-1]['label']='heldout'
meta=data['camera_metadata']['factory_camera_param']
intr=meta['color_intrinsic']
K=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1]],float)
dist=np.array([meta['color_distortion'][k] for k in ('k1','k2','p1','p2','k3')],float)
rng=np.random.default_rng(13)

def image_and_depth(sample):
    frame=sample['camera']
    return cv2.imread(str(root/frame['color_file'])),np.load(root/frame['depth_file'])*frame['depth_scale_mm']

def lift(pixels,depth):
    pixels=np.asarray(pixels,float).reshape(-1,2)
    norm=cv2.undistortPoints(pixels.reshape(-1,1,2),K,dist).reshape(-1,2)
    values=[]
    for x,y in pixels:
        u,v=int(round(x)),int(round(y))
        patch=depth[max(0,v-2):v+3,max(0,u-2):u+3]
        valid=patch[(patch>100)&(patch<3000)]
        values.append(float(np.median(valid)) if len(valid)>=8 else np.nan)
    z=np.array(values)
    return np.column_stack((norm[:,0]*z,norm[:,1]*z,z))

reference=data['samples'][0]
rgb0,depth0=image_and_depth(reference)
mask=np.zeros(rgb0.shape[:2],np.uint8)
mask[0:650,360:1110]=255
mask[200:720,490:800]=0 # exclude can and any moving foreground grip
orb=cv2.ORB_create(nfeatures=6500,fastThreshold=8)
kp0,des0=orb.detectAndCompute(rgb0,mask)
points0=lift([k.pt for k in kp0],depth0)
camera_poses=[];robot_poses=[];fit=[]
matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
for sample in data['samples']:
    if sample['index']==0:
        C=np.eye(4)
        stats={'label':sample['label'],'inliers':len(kp0),'reprojection_px':0,'depth_residual_mm':0}
    else:
        rgb,depth=image_and_depth(sample)
        kp,des=orb.detectAndCompute(rgb,None)
        pairs=matcher.knnMatch(des0,des,k=2)
        good=[m for pair in pairs if len(pair)==2 for m,n in [pair] if m.distance<0.7*n.distance and np.isfinite(points0[m.queryIdx]).all()]
        p0=np.array([points0[m.queryIdx] for m in good],float)
        uv=np.array([kp[m.trainIdx].pt for m in good],float)
        if len(good)<35:raise RuntimeError(f'{sample["label"]}: only {len(good)} valid matches')
        ok,rv,tv,inliers=cv2.solvePnPRansac(p0,uv,K,dist,iterationsCount=1500,reprojectionError=1.5,confidence=0.999,flags=cv2.SOLVEPNP_EPNP)
        if not ok or inliers is None or len(inliers)<30:raise RuntimeError(f'{sample["label"]}: PnP insufficient inliers')
        ids=inliers.reshape(-1)
        ok,rv,tv=cv2.solvePnP(p0[ids],uv[ids],K,dist,rv,tv,True,flags=cv2.SOLVEPNP_ITERATIVE)
        C=np.eye(4);C[:3,:3]=cv2.Rodrigues(rv)[0];C[:3,3]=tv.reshape(3)
        projected=cv2.projectPoints(p0[ids],rv,tv,K,dist)[0].reshape(-1,2)
        observed3=lift(uv[ids],depth)
        predicted3=p0[ids]@C[:3,:3].T+C[:3,3]
        valid=np.isfinite(observed3).all(axis=1)
        residual=np.linalg.norm(observed3[valid]-predicted3[valid],axis=1)
        stats={'label':sample['label'],'inliers':len(ids),
            'reprojection_px':float(np.median(np.linalg.norm(projected-uv[ids],axis=1))),
            'depth_residual_mm':float(np.median(residual)) if len(residual) else None}
    camera_poses.append(C);robot_poses.append(np.asarray(sample['T_base_tcp']))
    fit.append(stats)
    print(json.dumps(stats),flush=True)

train=[i for i,s in enumerate(data['samples']) if s['label'] not in ('heldout','returned')]
R,t=cv2.calibrateHandEye([robot_poses[i][:3,:3] for i in train],[robot_poses[i][:3,3] for i in train],
    [camera_poses[i][:3,:3] for i in train],[camera_poses[i][:3,3] for i in train],method=cv2.CALIB_HAND_EYE_PARK)
X=np.eye(4);X[:3,:3]=R;X[:3,3]=t.reshape(3)
if not np.isfinite(X).all() or np.linalg.norm(X[:3,3])>300:raise RuntimeError('Hand-eye solution nonphysical')
W=robot_poses[0]@X
consistency=[]
for sample,B,C in zip(data['samples'],robot_poses,camera_poses):
    expected=B@X@C
    delta=np.linalg.inv(W)@expected
    angle=np.degrees(np.arccos(np.clip((np.trace(delta[:3,:3])-1)/2,-1,1)))
    consistency.append({'label':sample['label'],'translation_mm':float(np.linalg.norm(delta[:3,3])),'angle_deg':float(angle)})

# Fit the foreground tabletop, excluding can, robot fingers and other containers.
ys,xs=np.mgrid[470:680:3,400:1120:3]
pixels=np.column_stack((xs.ravel(),ys.ravel()))
pixels=pixels[(pixels[:,0]<510)|(pixels[:,0]>790)]
points=lift(pixels,depth0)
points=points[np.isfinite(points).all(axis=1)]
points=points[(points[:,2]>130)&(points[:,2]<600)]
best=None
for _ in range(700):
    a,b,c=points[rng.choice(len(points),3,replace=False)]
    n=np.cross(b-a,c-a);length=np.linalg.norm(n)
    if length<1:continue
    n/=length;d=-n@a
    ids=np.abs(points@n+d)<3
    if best is None or ids.sum()>best.sum():best=ids
table=points[best];center=table.mean(axis=0)
n=np.linalg.svd(table-center)[2][-1]
if (W[:3,:3]@n)[2]<0:n=-n
d=-n@center
table_residual=float(np.median(np.abs(table@n+d)))
n_base=W[:3,:3]@n;d_base=float(d-n_base@W[:3,3])

# Green cylinder surface; axis constrained by table normal. Size comes from RGB-D, not SKU lookup.
hsv=cv2.cvtColor(rgb0,cv2.COLOR_BGR2HSV)
green=cv2.inRange(hsv,np.array([35,60,40]),np.array([100,255,255]))
roi=np.zeros_like(green);roi[230:710,490:800]=255;green=cv2.bitwise_and(green,roi)
y,x=np.where(green>0)
can_points=lift(np.column_stack((x[::3],y[::3])),depth0)
can_points=can_points[np.isfinite(can_points).all(axis=1)]
height=can_points@n+d
can_points=can_points[(height>0)&(height<250)&(can_points[:,2]<400)]
height=can_points@n+d
u=np.array([1.,0,0]);u-=n*(u@n);u/=np.linalg.norm(u);v=np.cross(n,u)
xy=np.column_stack((can_points@u,can_points@v))
best_circle=None;best_ids=None
for _ in range(1200):
    sample=xy[rng.choice(len(xy),3,replace=False)]
    A=np.column_stack((2*sample,np.ones(3)));rhs=np.sum(sample**2,axis=1)
    try:cx,cy,c0=np.linalg.solve(A,rhs)
    except np.linalg.LinAlgError:continue
    r2=c0+cx*cx+cy*cy
    if not 20**2<r2<45**2:continue
    radius=np.sqrt(r2);ids=np.abs(np.linalg.norm(xy-[cx,cy],axis=1)-radius)<3
    if best_ids is None or ids.sum()>best_ids.sum():best_ids=ids;best_circle=(cx,cy,radius)
if best_circle is None:raise RuntimeError('Cylinder fitting failed')
xy_in=xy[best_ids];A=np.column_stack((2*xy_in,np.ones(len(xy_in))))
cx,cy,c0=np.linalg.lstsq(A,np.sum(xy_in**2,axis=1),rcond=None)[0]
radius=float(np.sqrt(c0+cx*cx+cy*cy))
can_center_camera=cx*u+cy*v-d*n
bottom_base=W[:3,:3]@can_center_camera+W[:3,3]
height_est=float(np.percentile(height[best_ids],99)+5) # allowance for silver rim outside green mask
circle_error=float(np.median(np.abs(np.linalg.norm(xy_in-[cx,cy],axis=1)-radius)))
held=[r for r in consistency if r['label']=='heldout'][0]
accept=(held['translation_mm']<10 and held['angle_deg']<1.5 and table_residual<3
    and abs(n_base[2])>0.97 and 45<2*radius<85 and 90<height_est<210
    and all(r['translation_mm']<12 and r['angle_deg']<2 for r in consistency))
result={'schema_version':1,'source_run':str(root.resolve()),'status':'estimated_validated' if accept else 'rejected',
    'T_tcp_camera':X.tolist(),'T_base_camera_reference':W.tolist(),'feature_fits':fit,'hand_eye_consistency':consistency,
    'table_plane_base':{'normal':n_base.tolist(),'d_mm':d_base,'median_residual_mm':table_residual,'support_points':len(table)},
    'can':{'bottom_center_base_mm':bottom_base.tolist(),'axis_base':n_base.tolist(),
      'height_estimate_mm':height_est,'height_allowance_mm':10,'diameter_estimate_mm':2*radius,
      'circle_residual_mm':circle_error,'support_points':int(best_ids.sum()),'source':'local RGB-D green-cylinder fit'},
    'limitations':['Unknown hardware timestamps: captured only at stationary poses',
      'Mount collision mesh still conservative; excludes cable per user request',
      'Green-mask height includes estimated silver-rim allowance; not exact metrology']}
output=Path('calibration/geometry_estimate.json')
output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':result['status'],'T_tcp_camera':X.tolist(),'heldout':held,'table':result['table_plane_base'],'can':result['can']},ensure_ascii=False,indent=2))
raise SystemExit(0 if accept else 2)
