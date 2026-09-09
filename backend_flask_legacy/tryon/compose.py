from __future__ import annotations
from typing import Optional, Tuple
import cv2
import numpy as np
from .pose import PoseResult

_ARM_CHAINS = (("left_shoulder","left_elbow","left_wrist"),("right_shoulder","right_elbow","right_wrist"))

def _point(pose: PoseResult, name: str, vis: float = 0.30):
    try: p = pose.point(name)
    except Exception: return None
    if p is None: return None
    try:
        if pose.vis(name) < vis: return None
    except Exception: pass
    return np.asarray(p, dtype=np.float32)

def _resize(mask, frame_size):
    if mask is None: return None
    w,h = frame_size
    m=np.asarray(mask,dtype=np.float32)
    if m.ndim==3: m=m[:,:,0]
    if m.shape[:2]!=(h,w): m=cv2.resize(m,(w,h),interpolation=cv2.INTER_LINEAR)
    return np.clip(m,0,1)

def _segmentation(pose, frame_size):
    try: seg=pose.segmentation
    except Exception: seg=None
    return _resize(seg,frame_size)

def alpha_composite(background_bgr, overlay_bgra):
    if overlay_bgra is None: return background_bgr
    if overlay_bgra.shape[:2]!=background_bgr.shape[:2]: raise ValueError("overlay and frame sizes do not match")
    bg=background_bgr.astype(np.float32); ov=overlay_bgra.astype(np.float32)
    a=ov[:,:,3:4]/255.0
    return np.clip(a*ov[:,:,:3]+(1-a)*bg,0,255).astype(np.uint8)

def skin_tone_underlay(frame_bgr, pose, clothing_mask, overlay_bgra):
    """Recolour only uncovered replacement gaps using sampled skin chroma.

    Luminance comes from the original pixels so torso shading and wrinkles are
    retained; only Lab chroma is moved toward the user's visible skin.  The
    selected garment remains the top layer and normally covers most of this.
    """
    if clothing_mask is None or overlay_bgra is None:return frame_bgr
    h,w=frame_bgr.shape[:2]; m=np.asarray(clothing_mask,np.float32)
    if m.shape[:2]!=(h,w):m=cv2.resize(m,(w,h),interpolation=cv2.INTER_LINEAR)
    try:sw=pose.shoulder_width_px() or w*.25
    except Exception:sw=w*.25
    garment_alpha=overlay_bgra[:,:,3].astype(np.float32)/255.
    # Replace only old-clothing pixels that are genuinely left uncovered.  A
    # small protected rim prevents the artificial skin colour from producing
    # brown shoulder/collar halos around the selected garment.
    margin=max(5,int(sw*.045)); kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*margin+1,2*margin+1))
    protected=cv2.dilate((garment_alpha*255).astype(np.uint8),kernel).astype(np.float32)/255.
    replacement=np.clip(m*(1.-protected),0,1)
    if float(replacement.max())<.02:return frame_bgr

    samples=[]
    radius=max(5,int(sw*.045))
    for name in ("nose","left_wrist","right_wrist"):
        p=_point(pose,name,.20)
        if p is None:continue
        x,y=np.round(p).astype(int); x0=max(0,x-radius);x1=min(w,x+radius+1);y0=max(0,y-radius);y1=min(h,y+radius+1)
        if x1>x0 and y1>y0:samples.append(frame_bgr[y0:y1,x0:x1].reshape(-1,3))
    if not samples:return frame_bgr
    pixels=np.concatenate(samples,axis=0).astype(np.uint8).reshape(-1,1,3)
    sample_lab=cv2.cvtColor(pixels,cv2.COLOR_BGR2LAB).reshape(-1,3)
    # Ignore near-black/overexposed pixels and keep robust median chroma.
    valid=sample_lab[(sample_lab[:,0]>18)&(sample_lab[:,0]<238)]
    if len(valid)<10:return frame_bgr
    skin_ab=np.median(valid[:,1:3],axis=0)

    lab=cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2LAB).astype(np.float32)
    toned=lab.copy()
    # Suppress small fabric folds while keeping broad body lighting.
    toned[:,:,0]=cv2.GaussianBlur(lab[:,:,0],(0,0),sigmaX=max(1.,sw*.018))
    toned[:,:,1]=skin_ab[0]; toned[:,:,2]=skin_ab[1]
    toned_bgr=cv2.cvtColor(np.clip(toned,0,255).astype(np.uint8),cv2.COLOR_LAB2BGR)
    soft=cv2.GaussianBlur(replacement,(0,0),sigmaX=max(1.,sw*.012))[:,:,None]
    return np.clip(frame_bgr.astype(np.float32)*(1-soft)+toned_bgr.astype(np.float32)*soft,0,255).astype(np.uint8)

def blend_overlays(a,b,weight):
    if a is None:return b
    if b is None:return a
    weight=float(np.clip(weight,0,1))
    if weight<=.001:return a
    if weight>=.999:return b
    af=a.astype(np.float32); bf=b.astype(np.float32)
    aa=af[:,:,3:4]/255.; ba=bf[:,:,3:4]/255.
    ao=(1-weight)*aa+weight*ba
    premul=(1-weight)*af[:,:,:3]*aa+weight*bf[:,:,:3]*ba
    rgb=np.divide(premul,np.maximum(ao,1e-6))
    out=np.zeros_like(af); out[:,:,:3]=np.clip(rgb,0,255); out[:,:,3]=np.clip(ao[:,:,0]*255,0,255)
    return out.astype(np.uint8)

def _capsule(mask,p0,p1,radius,value=255):
    a=tuple(np.round(p0).astype(np.int32)); b=tuple(np.round(p1).astype(np.int32)); r=max(1,int(radius))
    cv2.line(mask,a,b,value,2*r); cv2.circle(mask,a,r,value,-1); cv2.circle(mask,b,r,value,-1)

def torso_clothing_mask(pose: PoseResult, frame_size: Tuple[int,int]) -> np.ndarray:
    """Body-shaped replacement mask. Interior is opaque; only its edge is soft."""
    w,h=frame_size; empty=np.zeros((h,w),np.float32)
    ls=_point(pose,"left_shoulder",.35); rs=_point(pose,"right_shoulder",.35)
    if ls is None or rs is None:return empty
    lh=_point(pose,"left_hip",.25); rh=_point(pose,"right_hip",.25)
    le=_point(pose,"left_elbow",.25); re=_point(pose,"right_elbow",.25); neck=_point(pose,"neck",.15)
    sv=rs-ls; sw=float(np.linalg.norm(sv))
    if sw<15:return empty
    sd=sv/sw; down=np.array([-sd[1],sd[0]],np.float32)
    if down[1]<0:down=-down
    if lh is None:lh=ls+down*sw*1.48
    if rh is None:rh=rs+down*sw*1.48
    hc=(lh+rh)*.5; hipw=float(np.linalg.norm(rh-lh))
    if hipw<sw*.35:
        hipw=sw*.82; lh=hc-sd*hipw*.5; rh=hc+sd*hipw*.5
    # Start above the shoulder line so the asset's real collar survives the
    # replacement mask. Starting below it slices the neckline into a straight
    # horizontal edge.
    sc=(ls+rs)*.5; top=sc-down*sw*.13; topw=sw*.78
    hem=hc-down*sw*.12; hemw=max(hipw*.90,sw*.88)
    hard=np.zeros((h,w),np.uint8); left=[]; right=[]
    levels=np.linspace(0,1,7); scales=(1,.985,.965,.95,.96,.985,1)
    for t,scale in zip(levels,scales):
        c=top*(1-t)+hem*t; width=(topw*(1-t)+hemw*t)*scale
        left.append(c-sd*width*.5); right.append(c+sd*width*.5)
    poly=np.array(left+right[::-1],np.float32); poly[:,0]=np.clip(poly[:,0],0,w-1); poly[:,1]=np.clip(poly[:,1],0,h-1)
    cv2.fillPoly(hard,[np.round(poly).astype(np.int32)],255)
    for shoulder,elbow in ((ls,le),(rs,re)):
        if elbow is None:continue
        d=elbow-shoulder; ln=float(np.linalg.norm(d))
        if ln<12:continue
        d/=ln; perp=np.array([-d[1],d[0]],np.float32); end=shoulder+d*min(ln*.48,sw*.52)
        r0=sw*.145; r1=sw*.095
        sp=np.array([shoulder+perp*r0,end+perp*r1,end-perp*r1,shoulder-perp*r0],np.float32)
        sp[:,0]=np.clip(sp[:,0],0,w-1); sp[:,1]=np.clip(sp[:,1],0,h-1)
        cv2.fillConvexPoly(hard,np.round(sp).astype(np.int32),255); cv2.circle(hard,tuple(np.round(shoulder).astype(np.int32)),max(3,int(r0)),255,-1)
    if neck is None: neck=sc+down*sw*.09
    # Keep the cutout conservative.  A large ellipse amputates the garment's
    # collar and creates the characteristic pasted-on U shape.  The asset's
    # own alpha supplies the precise neckline; this mask only guarantees skin
    # is not covered if an asset has a slightly dirty collar interior.
    collar_center=neck-down*sw*.025
    cv2.ellipse(hard,tuple(np.round(collar_center).astype(np.int32)),(max(6,int(sw*.105)),max(4,int(sw*.050))),0,0,360,0,-1)
    cut=max(4,int(sw*.075))
    for shoulder,elbow,wrist in ((ls,le,_point(pose,"left_wrist",.20)),(rs,re,_point(pose,"right_wrist",.20))):
        if elbow is None:continue
        start=shoulder+(elbow-shoulder)*.62; end=elbow if wrist is None else elbow+(wrist-elbow)*.22
        _capsule(hard,start,end,cut,0)
    seg=_segmentation(pose,frame_size)
    if seg is not None:
        person=(seg>.30).astype(np.uint8)*255
        person=cv2.morphologyEx(person,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
        hard=cv2.bitwise_and(hard,person)
    interior=(hard>127).astype(np.float32)
    edge=cv2.GaussianBlur(hard.astype(np.float32)/255.,(0,0),sigmaX=max(1.,sw*.008))
    return np.clip(np.maximum(interior*.98,edge),0,1)

def apply_clothing_mask(overlay_bgra, clothing_mask, frame_size: Optional[Tuple[int,int]]=None):
    """Hide the old shirt by keeping the new garment opaque inside the replacement mask."""
    if overlay_bgra is None or clothing_mask is None:return overlay_bgra
    h,w=overlay_bgra.shape[:2]; m=np.asarray(clothing_mask,np.float32)
    if m.shape[:2]!=(h,w):m=cv2.resize(m,(w,h),interpolation=cv2.INTER_LINEAR)
    m=np.clip(m,0,1); out=overlay_bgra.copy(); alpha=out[:,:,3].astype(np.float32)/255.
    final=alpha*np.maximum((m>=.50).astype(np.float32),m*.95)
    out[:,:,3]=np.clip(final*255,0,255).astype(np.uint8); return out

def limb_occlusion_mask(pose,frame_size,min_visibility=.35,thickness_scale=.16):
    w,h=frame_size; mask=np.zeros((h,w),np.uint8)
    try: sw=pose.shoulder_width_px() or w*.25
    except Exception: sw=w*.25
    base=max(5,int(sw*thickness_scale))
    for chain in _ARM_CHAINS:
        pts=[]
        for name in chain:
            p=_point(pose,name,min_visibility)
            if p is None:break
            pts.append(p)
        if len(pts)<2:continue
        for i in range(len(pts)-1):_capsule(mask,pts[i],pts[i+1],max(4,int(base*(1-.28*i))))
    result=cv2.GaussianBlur(mask,(0,0),sigmaX=max(1.,base*.18)).astype(np.float32)/255.
    seg=_segmentation(pose,frame_size)
    if seg is not None:result*=seg
    return np.clip(result,0,1)

def arms_in_front(pose,margin_m=.015):
    try:neck=pose.world_point("neck")
    except Exception:neck=None
    if neck is None:return False,False
    z=float(neck[2]); result=[]
    for side in ("left","right"):
        zs=[]
        for joint in ("elbow","wrist"):
            try:p=pose.world_point(f"{side}_{joint}")
            except Exception:p=None
            if p is not None:zs.append(float(p[2]))
        result.append(bool(zs) and min(zs)<z-margin_m)
    return result[0],result[1]

def person_occlusion_mask(pose,frame_size,use_depth_gate=True):
    w,h=frame_size
    if not use_depth_gate:return limb_occlusion_mask(pose,frame_size)
    lf,rf=arms_in_front(pose)
    try:sw=pose.shoulder_width_px() or w*.25
    except Exception:sw=w*.25
    base=max(5,int(sw*.145)); mask=np.zeros((h,w),np.uint8)

    # The head and neck are always in front of an upper-body garment. Without
    # this depth layer, an oversized catalogue collar can paint over the lower
    # face when the garment is scaled to broad shoulders.
    neck=_point(pose,"neck",.20); nose=_point(pose,"nose",.20)
    if neck is not None and nose is not None:
        direction=nose-neck; length=float(np.linalg.norm(direction))
        if length>8:
            start=neck+direction*.42
            _capsule(mask,start,nose,max(6,int(sw*.090)))

    for side,chain,active in zip(("left","right"),_ARM_CHAINS,(lf,rf)):
        if not active:continue
        pts=[]
        for name in chain:
            p=_point(pose,name,.35)
            if p is None:break
            pts.append(p)
        if len(pts)<2:continue
        # Preserve the virtual sleeve over the upper arm. Occlusion begins at
        # the sleeve exit, then restores the real forearm and hand in front.
        shoulder,elbow=pts[0],pts[1]
        sleeve_exit=shoulder+(elbow-shoulder)*.56
        visible_chain=[sleeve_exit]+pts[1:]
        for i in range(len(visible_chain)-1):
            _capsule(mask,visible_chain[i],visible_chain[i+1],max(4,int(base*(.82-.20*i))))

        hand=[]
        for suffix in ("wrist","thumb","index","pinky"):
            point=_point(pose,f"{side}_{suffix}",.20)
            if point is not None:hand.append(point)
        if hand:
            hull=cv2.convexHull(np.round(np.asarray(hand)).astype(np.int32))
            cv2.fillConvexPoly(mask,hull,255)
            wrist=hand[0]
            cv2.circle(mask,tuple(np.round(wrist).astype(np.int32)),max(5,int(base*.58)),255,-1)
    result=cv2.GaussianBlur(mask,(0,0),sigmaX=max(1.,base*.16)).astype(np.float32)/255.
    seg=_segmentation(pose,frame_size)
    if seg is not None:result*=seg
    return np.clip(result,0,1)

def apply_occlusion(overlay_bgra,mask):
    if overlay_bgra is None or mask is None:return overlay_bgra
    h,w=overlay_bgra.shape[:2];m=np.asarray(mask,np.float32)
    if m.shape[:2]!=(h,w):m=cv2.resize(m,(w,h),interpolation=cv2.INTER_LINEAR)
    out=overlay_bgra.copy();out[:,:,3]=np.clip(out[:,:,3].astype(np.float32)*(1-np.clip(m,0,1)),0,255).astype(np.uint8);return out

def feather_alpha(overlay_bgra,radius=1):
    if overlay_bgra is None or radius<=0:return overlay_bgra
    out=overlay_bgra.copy();a=out[:,:,3].astype(np.float32)/255.
    er=cv2.erode((a*255).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))).astype(np.float32)/255.
    blur=cv2.GaussianBlur(a,(0,0),sigmaX=float(radius));edge=np.clip(a-er,0,1)
    out[:,:,3]=np.clip(er+edge*blur,0,1)*255;out[:,:,3]=out[:,:,3].astype(np.uint8);return out

def match_lighting(overlay_bgra,frame_bgr,strength=.10):
    if overlay_bgra is None:return overlay_bgra
    covered=overlay_bgra[:,:,3]>32
    if not covered.any():return overlay_bgra
    out=overlay_bgra.copy();g=float(cv2.cvtColor(out[:,:,:3],cv2.COLOR_BGR2GRAY)[covered].mean());s=float(cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2GRAY)[covered].mean())
    if g<1:return out
    gain=float(np.clip(1+strength*(s/g-1),.82,1.18))
    out[:,:,:3]=np.clip(out[:,:,:3].astype(np.float32)*gain,0,255).astype(np.uint8)

    # Add a subtle continuous side falloff. Unlike copying webcam pixels, this
    # preserves the garment texture and cannot introduce old-shirt seams, while
    # still giving the flat catalogue image a rounded-torso appearance.
    ys,xs=np.where(covered)
    centre=(float(xs.min())+float(xs.max()))*.5
    half=max(1.0,(float(xs.max())-float(xs.min()))*.5)
    x=np.arange(out.shape[1],dtype=np.float32)
    curvature=1.0-.065*np.clip(np.abs(x-centre)/half,0,1)**1.7
    shaded=np.clip(out[:,:,:3].astype(np.float32)*curvature[None,:,None],0,255).astype(np.uint8)
    out[:,:,:3]=np.where(covered[:,:,None],shaded,out[:,:,:3])
    return out
