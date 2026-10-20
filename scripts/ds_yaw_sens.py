"""偏航角灵敏度：NCC 随 psi 变化，量化偏航角不确定度。"""
import json, math
from pathlib import Path
import numpy as np, cv2
K = dict(fx=691.4324951171875, fy=691.7568359375, cx=643.6455078125, cy=358.8835144042969)
exec(open('scripts/ds_handeye.py', encoding='utf-8').read().split('def main()')[0])
manifest = json.loads(Path('G:/Flip Anything/logs/experiments/legacy/ds_calib_manifest.json').read_text(encoding='utf-8'))
d = json.loads(Path('G:/Flip Anything/logs/experiments/legacy/ds_handeye.json').read_text(encoding='utf-8'))
n_up = np.array(d['n_cam_per_frame'][2]); n_up = n_up/np.linalg.norm(n_up)
a_vec = np.cross(n_up, [0,0,1.0]); a_vec/=np.linalg.norm(a_vec); b_vec = np.cross(n_up, a_vec)
gray, depth, Rb, pb = [], [], [], []
for f in manifest:
    b = Path(f['run'])/f['frame']
    gray.append(cv2.cvtColor(cv2.imread(str(b)+'_color.png'), cv2.COLOR_BGR2GRAY).astype(np.float32))
    depth.append(np.load(str(b)+'_depth_raw.npy').astype(np.float32))
    T = pose_matrix(f['pose']); Rb.append(T[:3,:3]); pb.append(T[:3,3])
h, w = gray[0].shape
vv, uu = np.mgrid[0:h, 0:w]
dyn = np.zeros((h,w), bool)
for i, j in [(2,3),(4,10),(8,9)]:
    dyn |= np.abs(gray[i]-gray[j]) > 22
dyn = cv2.dilate(dyn.astype(np.uint8), np.ones((13,13),np.uint8)).astype(bool)
def samples(i):
    dd = depth[i]
    sel = (dd>150)&(dd<1000)&(vv<640)&(~dyn)
    sel &= (np.arange(h)[:,None]%4==0)&(np.arange(w)[None,:]%4==0)
    vi, ui = np.nonzero(sel); Z = dd[vi,ui]
    return np.stack([(ui-K['cx'])*Z/K['fx'],(vi-K['cy'])*Z/K['fy'],Z],1).astype(np.float32), gray[i][vi,ui]
S = [samples(i) for i in range(len(manifest))]
def score(i,j,t_rel):
    P,a = S[i]; Q = P + t_rel[None,:]; z = Q[:,2]; ok = z>60
    u = K['fx']*Q[:,0]/z+K['cx']; v = K['fy']*Q[:,1]/z+K['cy']
    ok &= (u>2)&(u<w-3)&(v>2)&(v<h-3)
    return -1.0 if ok.sum()<200 else ncc(a[ok], bilinear(gray[j],u[ok],v[ok]))
pairs = [(i,j) for i in range(len(manifest)) for j in range(i+1,len(manifest))
         if np.linalg.norm(Rb[i]-Rb[j])<0.01 and 1<np.linalg.norm(pb[j]-pb[i])<200]
s_ph = d['depth_scale_photometric']
print('psi(deg)  平均NCC')
for dpsi in np.arange(-10, 10.1, 1.0):
    psi = math.radians(d['yaw_deg'] + dpsi)
    c1 = math.cos(psi)*a_vec + math.sin(psi)*b_vec
    Rcb = np.stack([c1, np.cross(n_up,c1), n_up],1)
    v = np.mean([score(i,j, Rcb @ (s_ph*(pb[i]-pb[j]))) for i,j in pairs])
    print(f'{d["yaw_deg"]+dpsi:8.2f}  {v:8.4f}' + ('   <-- peak' if abs(dpsi)<1e-9 else ''))
