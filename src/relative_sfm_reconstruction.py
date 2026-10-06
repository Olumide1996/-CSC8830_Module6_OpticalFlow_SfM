from pathlib import Path
import json
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/'data'/'sfm'; OUT=ROOT/'outputs'/'sfm'; OUT.mkdir(parents=True,exist_ok=True)
IMGS=[DATA/f'view{i}.jpeg' for i in range(1,5)]
CAL=DATA/'camera_calibration.npz'
MAX_FEATURES=3000; RATIO=.75; RANSAC=3.0

def load():
    d=np.load(CAL); K=d['camera_matrix'].astype(float); dist=d['dist_coeffs'].astype(float); imgs=[cv2.imread(str(p)) for p in IMGS]
    if any(x is None for x in imgs): raise RuntimeError('Could not read one or more SFM images.')
    return K,dist,d['image_size'].astype(int),imgs

def pair(a,b,K):
    sift=cv2.SIFT_create(nfeatures=MAX_FEATURES); ga=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY); gb=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
    ka,da=sift.detectAndCompute(ga,None); kb,db=sift.detectAndCompute(gb,None)
    knn=cv2.BFMatcher().knnMatch(da,db,k=2); good=[m for p in knn if len(p)==2 for m,n in [p] if m.distance<RATIO*n.distance]
    pa=np.float32([ka[m.queryIdx].pt for m in good]); pb=np.float32([kb[m.trainIdx].pt for m in good])
    H,mask=cv2.findHomography(pa,pb,cv2.RANSAC,RANSAC); mask=mask.ravel().astype(bool); ia,ib=pa[mask],pb[mask]
    proj=cv2.perspectiveTransform(ia.reshape(-1,1,2),H).reshape(-1,2); e=np.linalg.norm(proj-ib,axis=1)
    return H,ia,ib,len(good),int(mask.sum()),float(e.mean()),float(np.median(e))

def decomp(H,K):
    _,Rs,ts,ns=cv2.decomposeHomographyMat(H,K); return [{'R':R.astype(float),'t':t.reshape(3).astype(float),'n':n.reshape(3).astype(float),'solution':i+1} for i,(R,t,n) in enumerate(zip(Rs,ts,ns))]

def rays(p,K):
    q=np.c_[p,np.ones(len(p))]; return (np.linalg.inv(K)@q.T).T

def score(c,pa,K):
    r=rays(pa,K); den=r@c['n']; ok=np.abs(den)>1e-8
    if not ok.any(): return 0.,0,0
    X=r[ok]/den[ok,None]; X2=(c['R']@X.T).T+c['t']; pos=(X[:,2]>1e-8)&(X2[:,2]>1e-8); return float(pos.mean()),int(pos.sum()),len(pos)

def center(R,t): return (-R.T@t.reshape(3,1)).reshape(3)

def main():
    K,dist,size,imgs=load(); print('Camera matrix:\n',K); print('Image size:',size.tolist())
    pairs=[]
    for i in range(3):
        H,pa,pb,gm,inn,err,med=pair(imgs[i],imgs[i+1],K); print(f'\nView {i+1} -> {i+2}: {inn}/{gm} inliers ({100*inn/gm:.2f}%), error={err:.4f} px'); pairs.append((H,pa,pb,gm,inn,err,med))
    selected=[]; dec=[]
    for i,(H,pa,pb,*_) in enumerate(pairs,1):
        cs=decomp(H,K); scored=[(c,*score(c,pa,K)) for c in cs]; best=max(scored,key=lambda x:x[1]); c,frac,cnt,total=best; print(f'\nView {i} -> {i+1} decomposition:'); [print(f"  Solution {x[0]['solution']}: positive depth {100*x[1]:.2f}% ({x[2]}/{x[3]})") for x in scored]; print('Selected:',c['solution']); print('R=\n',c['R']); print('t=',c['t']); print('n=',c['n']); selected.append(c); dec.append({'pair':f'view{i}_to_view{i+1}','selected_solution':c['solution'],'positive_depth_fraction':frac,'positive_depth_count':cnt,'tested_points':total,'rotation':c['R'].tolist(),'translation_direction':c['t'].tolist(),'plane_normal':c['n'].tolist()})
    Rw=np.eye(3); tw=np.zeros(3); poses=[{'view':1,'camera_center_sfm_units':[0.,0.,0.]}]
    centers=[np.zeros(3)]
    for i,c in enumerate(selected,2):
        Rw=c['R']@Rw; tw=c['R']@tw+c['t']; C=center(Rw,tw); centers.append(C); poses.append({'view':i,'camera_center_sfm_units':C.tolist()}); print(f'View {i} camera center: {C}')
    results={'description':'Four-view planar monocular SFM using calibrated homography decomposition','scale_statement':'Translation scale is arbitrary; positions are relative SFM units, not millimeters.','reference_camera':'View 1','camera_matrix':K.tolist(),'distortion_coefficients':dist.tolist(),'image_size':size.tolist(),'homography_pairs':[{'pair':f'view{i+1}_to_view{i+2}','good_matches':p[3],'inliers':p[4],'inlier_ratio':p[4]/p[3],'mean_reprojection_error_px':p[5],'median_reprojection_error_px':p[6],'homography':p[0].tolist()} for i,p in enumerate(pairs)],'decomposition':dec,'camera_poses':poses}
    with open(OUT/'relative_sfm_results.json','w',encoding='utf-8') as f: json.dump(results,f,indent=4)
    with open(OUT/'relative_camera_positions.csv','w',encoding='utf-8') as f:
        f.write('view,x_sfm_units,y_sfm_units,z_sfm_units\n'); [f.write(f"{p['view']},{p['camera_center_sfm_units'][0]},{p['camera_center_sfm_units'][1]},{p['camera_center_sfm_units'][2]}\n") for p in poses]
    print('\nSaved:',OUT/'relative_sfm_results.json'); print('Saved:',OUT/'relative_camera_positions.csv')

if __name__=='__main__': main()
