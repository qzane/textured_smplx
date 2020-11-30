# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 01:43:32 2020

@author: qzane
"""

import os,sys
import shutil
import textured_smplx

USAGE = """python %s data_path, front_img, back_img, [model]
    data_path: the path to the data, should be like:
                data_path/images/XXX.jpg # image path
                data_path/smpl # output from smplify-x for SMPL model
                data_path/smplx # output from smplify-x for SMPLX model            
                data_path/images/XXX_PGN.jpg # PGN segmentation result (optional)
    front_img: name for the front image
    back_img: name for the back image
    model: can be 'smpl' or 'smplx', default
"""


def main(data_path, front_img, back_img, model='smpl'):
    ''' data_path: the path to the data, should be like:
            data_path/images/XXX.jpg # image path
            data_path/smpl # output from smplify-x for SMPL model
            data_path/smplx # output from smplify-x for SMPLX model            
            data_path/images/XXX_PGN.jpg # PGN segmentation result (optional)
        
        front_img: name for the front image
        back_img: name for the back image
        model: can be 'smpl' or 'smplx'
        
        return: texture will be data_path/texture_smpl.png or data_path/texture_smplx.png
    '''
    
    # step.0: check all the input data
    
    front_img = os.path.split(front_img)[-1] # remove the path
    back_img = os.path.split(back_img)[-1] # remove the path
    
    tmp = front_img.rfind('.')
    front_id = front_img[:tmp]
    tmp = back_img.rfind('.')
    back_id = back_img[:tmp]
    
    if model == 'smpl':
        template_obj = 'models/smpl_uv.obj'
        template_mask = 'models/smpl_mask_1000.png'
    elif model == 'smplx':
        template_obj = 'models/smplx_uv.obj'
        template_mask = 'models/smplx_mask_1000.png'
    else:
        raise(Exception("model type not found"))
        
    if not os.path.isfile(template_obj) or not os.path.isfile(template_mask):
        raise(Exception("model not found"))
        
    f_img = os.path.join(data_path, 'images', front_img)
    f_obj = os.path.join(data_path, model, 'meshes', front_id, '000.obj')
    f_pkl = os.path.join(data_path, model, 'results', front_id, '000.pkl')
    f_pgn = os.path.join(data_path, 'PGN', '%s_PGN.png'%front_id)
    if not os.path.isfile(f_pgn):
        f_pgn = None
    for fname, ftype in zip([f_img, f_obj, f_pkl], ['image','obj','pkl']):
        if not os.path.isfile(fname):
            raise(Exception("%d file for the front is not found"%ftype))
            
    b_img = os.path.join(data_path, 'images', back_img)
    b_obj = os.path.join(data_path, model, 'meshes', back_id, '000.obj')
    b_pkl = os.path.join(data_path, model, 'results', back_id, '000.pkl')
    b_pgn = os.path.join(data_path, 'PGN', '%s_PGN.png'%back_id)
    if not os.path.isfile(b_pgn):
        b_pgn = None
    for fname, ftype in zip([b_img, b_obj, b_pkl], ['image','obj','pkl']):
        if not os.path.isfile(fname):
            raise(Exception("%d file for the back is not found"%ftype))
            
            
    npath = os.path.join(data_path, 'texture_%s'%model)
    if f_pgn and b_pgn:
        pgn_path = os.path.join(data_path, 'PGN_%s'%model)
    else:
        pgn_path = None
        
    # step.1: produce single frame texture      
    textured_smplx.get_texture_SMPL(f_img, f_obj, f_pkl, npath, 'front', template_obj) 
    textured_smplx.get_texture_SMPL(b_img, b_obj, b_pkl, npath, 'back', template_obj)   
    
    # step.2: produce PGN texture (optional)
    
    textured_smplx.get_texture_SMPL(f_pgn, f_obj, f_pkl, npath, 'front_PGN', template_obj)
    textured_smplx.get_texture_SMPL(b_pgn, b_obj, b_pkl, npath, 'back_PGN', template_obj)
    
    # step3: combine all the textures
    
    textured_smplx.combine_texture_SMPL(npath)
    
    # step4: complete all the textures
    
    f_acc_texture = os.path.join(npath, 'back_texture_acc.png')
    f_acc_vis = os.path.join(npath, 'back_texture_vis_acc.png')
    f_mask = template_mask
    
    textured_smplx.complete_texture(f_acc_texture, f_acc_vis, f_mask)
    
    # finish: copy the result
    
    shutil.copyfile(os.path.join(npath, f_acc_texture[:-4]+'complete.png'),
                    os.path.join(data_path, 'texture_%s.png'%model)) 
    
    
if __name__ == '__main__':
    if len(sys.argv)<4:
        print(USAGE%sys.argv[0])
    elif len(sys.argv)==4:
        main(*sys.argv[1:4])
    else:
        main(*sys.argv[1:5])
        