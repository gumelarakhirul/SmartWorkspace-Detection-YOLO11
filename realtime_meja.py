"""Kamera Iriun: python realtime_meja.py. Q/Esc keluar, +/- confidence."""
import argparse
import os
import tempfile
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description='Object detection YOLO11 dengan Iriun.')
    parser.add_argument('--model',type=Path,default=BASE/'best.pt')
    parser.add_argument('--conf',type=float,default=0.40)
    parser.add_argument('--list-kamera',action='store_true')
    parser.add_argument('--kamera',choices=['iriun'],default='iriun')
    args=parser.parse_args()
    if not 0.40 <= args.conf <= 0.95:
        parser.error('--conf harus antara 0.40 dan 0.95.')
    import cv2
    from cv2_enumerate_cameras import enumerate_cameras
    if args.list_kamera:
        for c in enumerate_cameras(cv2.CAP_DSHOW):print(c.index,c.name)
        return
    if not args.model.is_file():parser.error('best.pt tidak ditemukan. Jalankan training dan cell simpan model di notebook.')
    import torch
    import numpy as np
    with tempfile.TemporaryDirectory(prefix='yolo11_camera_') as tmp:
        os.environ['YOLO_CONFIG_DIR']=tmp
        from ultralytics import YOLO
        DEVICE=0 if torch.cuda.is_available() else 'cpu'
        model=YOLO(str(args.model))
        expected=['keyboard','mouse','pc','monitor']
        if list(model.names.values()) != expected:
            parser.error('Model harus memiliki tepat empat kelas target. Jalankan training notebook.')
        CLASS_IDS=[cid for cid,name in model.names.items() if name in expected]
        print('Model:',args.model,'| Device:',DEVICE)
        print('Kelas:',[model.names[cid] for cid in CLASS_IDS])
        kamera_iriun(model, CLASS_IDS, DEVICE, args.conf)

def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    area_a=(a[2]-a[0])*(a[3]-a[1]);area_b=(b[2]-b[0])*(b[3]-b[1])
    return inter/max(area_a+area_b-inter,1e-9)

def kamera_iriun(model, CLASS_IDS, DEVICE, conf=0.40):
    import cv2
    import numpy as np
    from cv2_enumerate_cameras import enumerate_cameras
    cameras=list(enumerate_cameras(cv2.CAP_DSHOW))
    selected=next((c for c in cameras if 'iriun' in c.name.lower()),None)
    if selected is None:raise RuntimeError('Iriun tidak ditemukan. Buka aplikasi Iriun di PC/HP.')
    camera=cv2.VideoCapture(selected.index,cv2.CAP_DSHOW)
    window='YOLO11 - Object Detection (Iriun)'
    previous=[]
    try:
        if not camera.isOpened():raise RuntimeError('Kamera tidak dapat dibuka. Tutup aplikasi kamera lain.')
        cv2.namedWindow(window,cv2.WINDOW_NORMAL)
        while True:
            started=time.perf_counter();ok,frame=camera.read()
            if not ok:raise RuntimeError('Iriun tidak mengirim frame.')
            message='Deteksi objek';count=0
            if float(frame.mean())<2 and int(frame.max())<8:
                image=frame.copy();previous=[];message='Frame hitam: periksa Iriun'
            else:
                result=model.predict(frame,conf=conf,classes=CLASS_IDS,iou=.4,device=DEVICE,verbose=False)[0]
                current=[];keep=[];used=set()
                for i,(cid,box) in enumerate(zip(result.boxes.cls.tolist(),result.boxes.xyxy.tolist())):
                    matches=[(iou(box,b),j) for j,(k,b,n) in enumerate(previous) if k==cid and j not in used]
                    overlap,j=max(matches,default=(0,-1));n=previous[j][2]+1 if overlap>=.4 else 1
                    if overlap>=.4:used.add(j)
                    current.append((cid,box,n))
                    if n>=2:keep.append(i)
                previous=current;result=result[keep];count=len(result.boxes)
                aliases={'keyboard':'Keyboard','monitor':'Monitor','mouse':'Mouse','pc':'PC'}
                result.names={cid:aliases.get(name,name) for cid,name in model.names.items()}
                image=result.plot(line_width=2,font_size=14)
            fps=1/max(time.perf_counter()-started,1e-6)
            h,w=image.shape[:2];canvas=np.zeros((h+60,max(w,640),3),dtype=np.uint8);canvas[60:,:w]=image
            cv2.putText(canvas,message,(10,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            cv2.putText(canvas,f'Objek {count} | conf {conf:.2f} | FPS {fps:.1f} | +/- confidence | Q keluar',
                        (10,45),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            cv2.imshow(window,canvas);key=cv2.waitKey(1)&0xFF
            if key in (ord('q'),ord('Q'),27) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:break
            if key in (ord('-'),ord('_')):conf=max(.4,round(conf-.05,2));previous=[]
            if key in (ord('+'),ord('=')):conf=min(.95,round(conf+.05,2));previous=[]
    finally:
        camera.release();cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
