import numpy as np
import scipy
import scipy.optimize
import cv2



def draw_p2d(img, p2d, index=None, draw_id=False, p_size=1):
    ''' img: RGB image
        p2d: [[x,y]...] or [[x,y,z]...]
        index: only draw p2d[index]
        draw_id: todo
        return: new img
    '''
    if type(p2d) == list:
        p2d = np.array(p2d)
    p2d = p2d.squeeze()
    if len(p2d.shape) == 1:
        p2d = p2d.reshape(1, -1)
    assert(len(p2d.shape)==2)
    assert(p2d.shape[1] in (2,3))
    if p2d.shape[1] == 3:
        z = p2d[:,2]
        p2d = p2d[:,:2]
    else:
        z = np.ones(p2d.shape[0])
    colors = np.arange(0,25).astype(np.uint8)*10
    colors = cv2.applyColorMap(colors, cv2.COLORMAP_HSV)
    colors = colors.reshape(-1, 3).tolist()
    ncolors = len(colors)
    if index is None:
        index = range(len(p2d))
    img = img.copy()
    h,w = img.shape[:2]
    for no in index:
        x,y = p2d[no]
        x,y = int(x), int(y)
        if z[no]>0 and 0<=x<w and 0<=y<h:
            cv2.circle(img, (x,y), p_size, colors[no%ncolors], -1)
            if draw_id:
                img = cv2.putText(img, '%d'%no, (x+5,y+5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1, 
                      (0,0,255), 1)
    return img

def solve_Ax_B(A, B):
    return np.linalg.lstsq(A, B, rcond=-1)[0]
    
    
def obj_vv(fname): # read vertices: (x,y,z)
    res = []
    with open(fname) as f:
        for line in f:
            if line.startswith('v '):
                tmp = line.split(' ')
                v = [float(i) for i in tmp[1:4]]
                res.append(v)
                
    return np.array(res, dtype=np.float)
    
def obj_vt(fname): # read texture coordinates: (u,v)
    res = []
    with open(fname) as f:
        for line in f:
            if line.startswith('vt '):
                tmp = line.split(' ')
                v = [float(i) for i in tmp[1:3]]
                res.append(v)
    return np.array(res, dtype=np.float)

def obj_fv(fname): # read vertices id in faces: (vv1,vv2,vv3)
    res = []
    with open(fname) as f:
        for line in f:
            if line.startswith('f '):
                tmp = line.split(' ')
                if '/' in tmp[1]:
                    v = [int(i.split('/')[0]) for i in tmp[1:4]]
                else:
                    v = [int(i) for i in tmp[1:4]]
                res.append(v)
    return np.array(res, dtype=np.int) - 1 # obj index from 1

def obj_ft(fname): # read texture id in faces: (vt1,vt2,vt3)
    res = []
    with open(fname) as f:
        for line in f:
            if line.startswith('f '):
                tmp = line.split(' ')
                if '/' in tmp[1]:
                    v = [int(i.split('/')[1]) for i in tmp[1:4]]
                else:
                    raise(Exception("not a textured obj file"))
                res.append(v)
    return np.array(res, dtype=np.int) - 1 # obj index from 1


def fv2norm(fv, vv):
    ''' calculate face norm
    # similar to the following method using trimesh 
    # mesh = trimesh.Trimesh(vv, fv, process=False)
    # return mesh.face_normals
    '''    
    p1 = vv[fv[:,0]]
    p2 = vv[fv[:,1]]
    p3 = vv[fv[:,2]]
    p12 = p2 - p1
    p13 = p3 - p1
    n = np.cross(p12, p13)
    n = n / (np.sqrt((n*n).sum(1))).reshape(-1,1)
    fnorm = n
    return fnorm

def p3d_p2d(p3d, rmat, tvec, cameraMatrix, distCoeffs=None, img_shape=None):
    ''' similar to cv2.projectPoints, there are 3 difference:
        1. use rmat as input rather than rvec
        2. the output is [[u,v,z]...], when z>0, the point is visible
        3. by providing img_shape, you can get rid of invisible points due to high distortion
    '''
    p_w = p3d.copy().T # world coordinate
    p_c_0 = rmat.dot(p_w) + tvec.reshape(3, 1) # camera coordinate
    z = p_c_0[2, :].copy().reshape(1, -1) # z (>0 are visible points)
    p_c_1 = p_c_0[:2] / z # camera coordinate (normalized)
    
    if img_shape:
        p_c_dis = np.ones_like(p_c_0) # undistort coordinates
        p_c_dis[:2,:] = p_c_1       
        p_uv = cameraMatrix.dot(p_c_dis)
        h,w = img_shape
        z[0,p_uv[0,:]<0]=-1
        z[0,p_uv[0,:]>w]=-1
        z[0,p_uv[1,:]<0]=-1
        z[0,p_uv[1,:]>h]=-1
    
    if distCoeffs is not None:
        k1,k2,p1,p2,k3 = distCoeffs
        r2 = (p_c_1 * p_c_1).sum(0).copy()
        kv = (1+k1*r2+k2*r2*r2+k3*r2*r2*r2)
        xy = (p_c_1[0,:] * p_c_1[1,:]).copy()
        x2 = p_c_1[0,:] * p_c_1[0,:]
        y2 = p_c_1[1,:] * p_c_1[1,:]
        p_c_1[0,:] = p_c_1[0,:]*kv + 2*p1*xy + p2*(r2+2*x2)
        p_c_1[1,:] = p_c_1[1,:]*kv + p1*(r2+2*y2) + 2*p2*xy
    p_c_dis = np.ones_like(p_c_0) # undistort coordinates
    p_c_dis[:2,:] = p_c_1       
    p_uv = cameraMatrix.dot(p_c_dis)
    p_uv[2,:] = z
    return p_uv.T

def cv2_triangle(img, p123):
    ''' draw triangles using OpenCV '''
    p1, p2, p3 = (tuple(i) for i in p123)
    cv2.line(img, p1, p2, (255, 0, 0), 1) 
    cv2.line(img, p2, p3, (255, 0, 0), 1) 
    cv2.line(img, p1, p3, (255, 0, 0), 1)
    return img
