"""Pixel-exact crop rectangle and original-image patch coordinates; no model imports."""
import math


def crop_rectangle(subject,obj,width,height):
    assert width>0 and height>0
    for b in (subject,obj):assert len(b)==4 and 0<=b[0]<b[2]<=1 and 0<=b[1]<b[3]<=1
    x0=min(subject[0],obj[0])*width;y0=min(subject[1],obj[1])*height
    x1=max(subject[2],obj[2])*width;y1=max(subject[3],obj[3])*height
    dx=.1*(x1-x0);dy=.1*(y1-y0)
    return [max(0,math.floor(x0-dx)),max(0,math.floor(y0-dy)),min(width,math.ceil(x1+dx)),min(height,math.ceil(y1+dy))]


def patch_centers(rect,width,height):
    left,top,right,bottom=rect
    assert 0<=left<right<=width and 0<=top<bottom<=height
    return [[(left+(c+.5)*(right-left)/14)/width,(top+(r+.5)*(bottom-top)/14)/height] for r in range(14) for c in range(14)]
