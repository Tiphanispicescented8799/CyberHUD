"""Original cyan/amber interface drawn entirely with OpenCV primitives."""
import cv2
from .geometry import bracket_segments
from .effects import pulse, scan_y

CYAN = (230, 225, 65)
AMBER = (60, 175, 255)
MUTED = (115, 130, 135)


def text(frame, message, x, y, color=CYAN, scale=0.43):
    height, width = frame.shape[:2]
    size, _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = max(2, min(int(x), width-size[0]-3))
    y = max(size[1]+2, min(int(y), height-4))
    for thickness, ink in ((3, (8,12,15)), (1, color)):
        cv2.putText(frame, message, (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, ink, thickness, cv2.LINE_AA)


class HUD:
    def draw(self, frame, targets, now, fps, timestamp, dropped=False):
        from .tracking import tracking_status
        h, w = frame.shape[:2]
        compact = w < 650
        for band in (frame[:62], frame[max(0,h-58):]):
            cv2.convertScaleAbs(band, dst=band, alpha=0.42)
        text(frame, 'CYBERHUD', 18, 27, scale=0.67 if not compact else 0.5)
        text(frame, 'V2 / LOCAL VISION', 19, 48, MUTED, 0.34)
        text(frame, f'{fps:04.1f} FPS', w-113, 26, scale=0.46)
        if not compact:
            text(frame, timestamp, w-212, 48, MUTED, 0.4)
            cv2.line(frame,(190,20),(min(w-230,360),20),MUTED,1)
            cv2.circle(frame,(200,20),3,CYAN,-1)
        # Minimal corner guides frame the camera without a full rectangular border.
        for start,end in bracket_segments((10,70,w-11,h-68),22):
            cv2.line(frame,start,end,MUTED,1,cv2.LINE_AA)
        for target in targets:
            lost = target.state == 'LOST'
            color = AMBER if lost or target.state == 'ACQUIRING' else CYAN
            acquiring = target.state == 'ACQUIRING'
            progress = min(1,max(0,(now-target.acquired_at)/0.3))
            pad = 9 + round((1-progress)*18 if acquiring else 2*pulse(now))
            bounds = target.box.clipped(w,h,pad)
            x1,y1,x2,y2 = bounds
            length = 16+round(4*pulse(now))
            for start,end in bracket_segments(bounds,length):
                cv2.line(frame,start,end,tuple(int(c*0.22) for c in color),4,cv2.LINE_AA)
                cv2.line(frame,start,end,color,1 if lost else 2,cv2.LINE_AA)
            cx,cy = (round(v) for v in target.box.center)
            radius = 10
            if target.state == 'LOCKED' and target.locked_at is not None:
                radius += round(18*max(0,1-(now-target.locked_at)/0.5))
            angle = int(now*38)%360
            for offset in (0,180):
                cv2.ellipse(frame,(cx,cy),(radius,radius),angle+offset,15,140,color,1,cv2.LINE_AA)
            cv2.circle(frame,(cx,cy),2,color,-1,cv2.LINE_AA)
            cv2.line(frame,(cx-19,cy),(cx-14,cy),color,1)
            cv2.line(frame,(cx+14,cy),(cx+19,cy),color,1)
            if not lost:
                sy = scan_y(y1,y2,now+target.id*0.19,3.0)
                cv2.line(frame,(x1+2,sy),(x2-2,sy),tuple(int(c*0.55) for c in color),1,cv2.LINE_AA)
                cv2.line(frame,(x1,sy-3),(x1,sy+3),color,2)
            label = 'LOST / REACQUIRING' if lost else target.state
            if acquiring:
                label += '...'
            label_y = max(88,y1-25)
            text(frame,target.label,x1,label_y,color,0.46)
            text(frame,label,x1,label_y+16,color,0.34)
            text(frame,f'X {cx:04d} / Y {cy:04d}',x1,min(h-76,y2+17),MUTED,0.34)
            # This is the last actual model score, not tracking certainty.
            if target.confidence is not None:
                score = max(0,min(1,target.confidence))
                by = min(h-69,y2+26)
                end = min(w-3,x1+55)
                cv2.line(frame,(x1,by),(end,by),MUTED,2)
                cv2.line(frame,(x1,by),(x1+round((end-x1)*score),by),color,2)
                text(frame,f'LAST DET {score:.2f}',end+7,by+4,MUTED,0.3)
        headline,state,count = tracking_status(targets)
        lost_count = sum(t.state == 'LOST' for t in targets)
        if dropped:
            headline = 'CAMERA SIGNAL LOST / RETRYING'
        text(frame,headline,18,h-39,AMBER if dropped or state == 'LOST' else CYAN,0.39 if compact else 0.46)
        detail = f'TARGETS {count:02d} / {state}'
        if lost_count:
            detail += f' / LOST {lost_count:02d}'
        text(frame,detail,18,h-20,MUTED,0.34 if compact else 0.4)
        if not compact:
            text(frame,f'FRAME {w}x{h} / Q QUIT',w-225,h-20,MUTED,0.37)
        return frame
