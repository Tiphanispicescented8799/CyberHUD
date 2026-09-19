"""Time-based animation independent of camera frame rate."""
from math import sin, pi


def pulse(now):
    return (sin(now*2*pi*0.8)+1)/2


def scan_y(top, bottom, now, period=2.4):
    phase = (now % period)/period
    return round(top+(bottom-top)*(1-abs(2*phase-1)))
