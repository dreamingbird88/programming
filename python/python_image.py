"""
sudo pip install Pillow

ipython2.7
run python_image.py

python python_image.py Claire_20161231_China

"""

from __future__ import print_function
from PIL import Image
import math
import numpy as np

def main(img_file):
  in_img = img_file

  # CN [33, 48] cm == [1.2992, 1.8898] inches
  size_cn = [1.2992, 1.8898]
  size_us = [2.0000, 2.0000]

  # input.thumbnail(size, Image.ANTIALIAS) # zoom the image.
  # print(input.format, input.size, input.mode)
  input = Image.open('%s.jpg' % in_img)

  country = ""
  img_size = input.size
  ratio = 1.0*img_size[0]/img_size[1]
  if abs(1.0 - ratio) < 0.001: country = "us"
  if abs(size_cn[0]/size_cn[1] - ratio) < 0.001: country = "cn"
  assert country != "", "Unmatched ratio %.3f." % ratio

  size = size_cn if country == "cn" else size_us
  dpi = int(math.ceil(max(img_size[0]/size[0], img_size[1]/size[1])))
  output = Image.new("RGB", [6*dpi, 4*dpi], color=(0,0,255))
  #output = Image.new("RGBA", [6*dpi, 4*dpi], color=(0,0,255,253)) # blue. alpha=opacity

  x,y0,d = (np.array((0.25, 0.12, 0.004))*dpi).astype(int)
  m,n = 4,2
  if country == "us":
    x,y0,d = 0,0,0
    m,n = 3,2
  for i in range(0, m):
    y = y0
    for j in range(0,n):
      output.paste(input,(x, y)) # also (x, y, x_end, y_end)
      y += input.size[1] + d
    x += input.size[0] + d
  output.save('%s_4x6.jpg'%in_img)

if __name__ == "__main__":
  import sys
  main(sys.argv[1])
