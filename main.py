import os
import sys
sys.path.insert(0, os.path.dirname(__file__) if '__file__' in globals() else '.')

from maix import camera, display, uart, app, image
import time
import mode_rect  # type: ignore
import mode_laser # type: ignore

CAM_W = 320
CAM_H = 240
FPS = 30
CAM_BUFFERS = 1
DISPLAY_FPS = 15
DISPLAY_INTERVAL = 1.0 / DISPLAY_FPS
FPS_EMA_ALPHA = 0.2

cam = camera.Camera(CAM_W, CAM_H, fps=FPS, buff_num=CAM_BUFFERS)
disp = display.Display()
serial_dev = uart.UART('/dev/ttyS0', 115200)

cam.exposure(500)

mode = 'rect'
mode_rect.init_rect()
next_display_time = time.monotonic()
last_frame_time = None
realtime_fps = 0.0


def interval_due(now, next_time, interval):
    if now < next_time:
        return False, next_time
    if now - next_time > interval:
        next_time = now
    return True, next_time + interval


print('--- main_switch started ---')

while not app.need_exit():
    data = serial_dev.read()
    if data:
        text = data.decode('utf-8', 'ignore')
        for char in text:
            if char == '1':
                mode = 'rect'
                mode_rect.init_rect()
                cam.exposure(200)
                print('switch -> rect')
            elif char == '2':
                mode = 'laser'
                mode_laser.init_laser()
                cam.exposure(400)
                print('switch -> laser')

    frame = cam.read()
    if frame is None:
        continue

    now = time.monotonic()
    if last_frame_time is not None:
        frame_interval = now - last_frame_time
        if frame_interval > 0:
            instant_fps = 1.0 / frame_interval
            if realtime_fps <= 0:
                realtime_fps = instant_fps
            else:
                realtime_fps += FPS_EMA_ALPHA * (instant_fps - realtime_fps)
    last_frame_time = now

    display_due, next_display_time = interval_due(now, next_display_time, DISPLAY_INTERVAL)
    frame_bgr = image.image2cv(frame, ensure_bgr=True, copy=False)

    if mode == 'rect':
        mode_rect.step_rect(frame_bgr, serial_dev, disp, display_due, realtime_fps)
    else:
        mode_laser.step_laser(frame_bgr, serial_dev, disp, display_due, realtime_fps)
