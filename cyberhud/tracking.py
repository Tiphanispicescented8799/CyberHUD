"""Geometric association only; IDs are temporary session counters."""
from dataclasses import dataclass
from math import exp, hypot
from .geometry import Box, iou


@dataclass
class Target:
    id: int
    box: Box
    measured: Box
    last_seen: float
    visible: bool = True
    confidence: float | None = None
    state: str = 'ACQUIRING'
    acquired_at: float = 0.0
    locked_at: float | None = None
    hits: int = 1
    velocity: tuple = (0.0, 0.0)

    @property
    def label(self):
        return f"TARGET-{self.id:02d}"


class Tracker:
    def __init__(self, ttl=1.6, smoothing=0.075, miss_grace=0.45):
        self.ttl = ttl
        self.miss_grace = min(miss_grace,ttl*0.75)
        self.smoothing = smoothing
        self.targets = []
        self.next_id = 1
        self.last_update = None

    def update(self, detections, now):
        dt = 1/30 if self.last_update is None else max(0, now-self.last_update)
        self.last_update = now
        self.targets = [t for t in self.targets if now-t.last_seen <= self.ttl]
        if detections is not None:
            boxes = [d if isinstance(d,Box) else d.box for d in detections]
            pairs = []
            for ti, target in enumerate(self.targets):
                target.visible = False
                age = min(now-target.last_seen,0.25)
                vx,vy = target.velocity
                predicted = Box(target.measured.x+vx*age,target.measured.y+vy*age,
                                target.measured.w,target.measured.h)
                for di, box in enumerate(boxes):
                    distance = hypot(*(a-b for a,b in zip(predicted.center, box.center)))
                    scale = max(target.measured.w, target.measured.h, box.w, box.h, 1)
                    overlap = iou(predicted, box)
                    ratio = (box.w*box.h)/max(1,target.measured.w*target.measured.h)
                    if 0.35 < ratio < 2.85 and (overlap > 0.12 or distance/scale < 0.8):
                        pairs.append((1-overlap+distance/scale, ti, di))
            used_t, used_d = set(), set()
            for _, ti, di in sorted(pairs):
                if ti in used_t or di in used_d:
                    continue
                target = self.targets[ti]
                interval = max(0.001,now-target.last_seen)
                if now-target.last_seen > self.miss_grace:
                    target.acquired_at = now
                    target.locked_at = None
                    target.hits = 0
                    target.velocity = (0.0,0.0)
                else:
                    limit = max(boxes[di].w,boxes[di].h)*4
                    target.velocity = tuple(0.5*old+0.5*max(-limit,min(limit,(b-a)/interval))
                        for old,a,b in zip(target.velocity,target.measured.center,boxes[di].center))
                target.measured = boxes[di]
                target.last_seen = now
                target.visible = True
                target.hits += 1
                target.confidence = getattr(detections[di],'confidence',None)
                used_t.add(ti)
                used_d.add(di)
            for di, box in enumerate(boxes):
                if di not in used_d:
                    self.targets.append(Target(self.next_id,box,box,now,
                        confidence=getattr(detections[di],'confidence',None),acquired_at=now))
                    self.next_id += 1
        alpha = 1-exp(-dt/self.smoothing) if self.smoothing > 0 else 1
        for target in self.targets:
            target.box = target.box.blend(target.measured, alpha)
            if now-target.last_seen > self.miss_grace:
                target.state = 'LOST'
            elif target.locked_at is None:
                if target.visible and target.hits >= 2 and now-target.acquired_at >= 0.15:
                    target.locked_at = now
                    target.state = 'LOCKED'
                else:
                    target.state = 'ACQUIRING'
            else:
                target.state = 'LOCKED' if now-target.locked_at < 0.5 else 'TRACKING'
        return self.targets


def tracking_status(targets):
    """Global status and count use the same retained targets as the face HUD."""
    if not targets:
        return 'SCANNING / NO TARGET', 'SCANNING', 0
    states = {target.state for target in targets}
    state = next(s for s in ('TRACKING','LOCKED','ACQUIRING','LOST') if s in states)
    return 'TARGET TRACKING // ACTIVE', state, len(targets)
