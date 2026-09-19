"""Pure geometry shared by detection, tracking and rendering."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def center(self):
        return self.x + self.w / 2, self.y + self.h / 2

    def blend(self, other, alpha):
        return Box(*(a + alpha * (b - a) for a, b in zip(
            (self.x, self.y, self.w, self.h),
            (other.x, other.y, other.w, other.h))))

    def clipped(self, width, height, padding=0):
        x1 = max(0, min(width - 1, round(self.x - padding)))
        y1 = max(0, min(height - 1, round(self.y - padding)))
        x2 = max(x1, min(width - 1, round(self.x + self.w + padding)))
        y2 = max(y1, min(height - 1, round(self.y + self.h + padding)))
        return x1, y1, x2, y2


def iou(a, b):
    overlap = max(0, min(a.x+a.w, b.x+b.w)-max(a.x, b.x)) * max(
        0, min(a.y+a.h, b.y+b.h)-max(a.y, b.y))
    union = a.w*a.h + b.w*b.h - overlap
    return overlap / union if union > 0 else 0.0


def bracket_segments(bounds, length):
    x1, y1, x2, y2 = bounds
    length = max(0, min(length, (x2-x1)//2, (y2-y1)//2))
    return [segment for x, y, dx, dy in (
        (x1,y1,1,1), (x2,y1,-1,1), (x1,y2,1,-1), (x2,y2,-1,-1))
        for segment in (((x,y),(x+dx*length,y)), ((x,y),(x,y+dy*length)))]
