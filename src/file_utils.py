"""Utilities for manipulating files.
"""
import os, PIL, shutil, cv2

exists   = lambda path: os.path.exists(path)
fopen    = lambda path, mode: open(path, mode)
makedirs = lambda path: os.makedirs(path)
listdir  = lambda path: os.listdir(path)
copyfile = lambda a, b: shutil.copyfile(a, b)

def write_image(image_path, rgb):
  ext = os.path.splitext(image_path)[1]
  with fopen(image_path, 'w') as f:
    img_str = cv2.imencode(ext, rgb[:,:,::-1])[1].tostring()
    f.write(img_str)

def read_image(image_path, type='rgb'):
  with fopen(file_name, 'r') as f:
    I = PIL.Image.open(f)
    II = np.array(I)
    if type == 'rgb':
      II = II[:,:,:3]
  return II
