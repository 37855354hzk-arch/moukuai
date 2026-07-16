import cv2
import numpy as np
import time
from maix import image

from hardware import shared_gpio as led # type: ignore

CAM_W, CAM_H = 320, 240
MAP_SEARCH_SIZE = 180
MAX_MAP_LOST = 2

MIN_OUTER_AREA = 1000
MIN_INNER_AREA = 600

ALIGN_THR_X = 8
ALIGN_THR_Y = 8

SEND_INTERVAL = 0.02

KERNEL_SMALL = np.ones((3, 3), np.uint8)
KERNEL_LARGE = np.ones((5, 5), np.uint8)

last_mid_center = None
last_mid_rect = None
lost_map_cnt = 0
tracking_active = False
open_sent = False
next_send_time = time.monotonic()


def init_rect():
    global last_mid_center, last_mid_rect, lost_map_cnt, tracking_active, open_sent, next_send_time
    last_mid_center = None
    last_mid_rect = None
    lost_map_cnt = 0
    tracking_active = False
    open_sent = False
    next_send_time = time.monotonic()
    led.value(0)
    print('--- rect mode initialized ---')


def send_due():
    global next_send_time
    now = time.monotonic()
    if now < next_send_time:
        return False
    if now - next_send_time > SEND_INTERVAL:
        next_send_time = now
    next_send_time += SEND_INTERVAL
    return True


def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def draw_quad(canvas, quad, color, thickness=2):
    quad_i = quad.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [quad_i], True, color, thickness)


def crop_around_center(img_bgr, center, size):
    h, w = img_bgr.shape[:2]
    if center is None:
        return img_bgr, (0, 0)

    cx, cy = center
    half = size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, x1 + size)
    y2 = min(h, y1 + size)

    x1 = max(0, x2 - size)
    y1 = max(0, y2 - size)
    return img_bgr[y1:y2, x1:x2], (x1, y1)


def detect_map_rects(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 7
    )

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, KERNEL_SMALL)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, KERNEL_LARGE)

    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None:
        return None

    hierarchy = hierarchy[0]
    best = None

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < MIN_OUTER_AREA:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        child_idx = hierarchy[i][2]
        if child_idx == -1:
            continue

        valid_inner_approx = None
        current_child_idx = child_idx
        while current_child_idx != -1:
            child = contours[current_child_idx]
            child_area = cv2.contourArea(child)

            if child_area >= MIN_INNER_AREA:
                child_peri = cv2.arcLength(child, True)
                child_approx = cv2.approxPolyDP(child, 0.03 * child_peri, True)

                if len(child_approx) == 4 and cv2.isContourConvex(child_approx):
                    ratio = child_area / float(area)
                    if 0.05 <= ratio <= 0.95:
                        valid_inner_approx = child_approx
                        break

            current_child_idx = hierarchy[current_child_idx][0]

        if valid_inner_approx is None:
            continue

        outer = order_points(approx.reshape(4, 2).astype(np.float32))
        inner = order_points(valid_inner_approx.reshape(4, 2).astype(np.float32))
        mid = (outer + inner) / 2.0

        if best is None or area > best['area']:
            best = {'outer': outer, 'inner': inner, 'mid': mid, 'area': area}

    return best


def step_rect(frame_bgr, serial_dev, disp, show=True, fps=0.0):
    global last_mid_center, last_mid_rect, lost_map_cnt, tracking_active, open_sent

    h, w = frame_bgr.shape[:2]
    frame_center = (w // 2, h // 2)

    map_input, (off_x, off_y) = crop_around_center(frame_bgr, last_mid_center, MAP_SEARCH_SIZE)
    map_result = detect_map_rects(map_input)

    status_text = 'SEARCHING'
    err_x = None
    err_y = None
    outer_rect = None
    inner_rect = None
    mid_rect = None

    if map_result is not None:
        offset = np.array([off_x, off_y], dtype=np.float32)
        outer_rect = map_result['outer'] + offset
        inner_rect = map_result['inner'] + offset
        mid_rect = map_result['mid'] + offset

        last_mid_rect = mid_rect
        last_mid_center = (
            int(mid_rect[:, 0].mean()),
            int(mid_rect[:, 1].mean())
        )
        lost_map_cnt = 0

        cx, cy = last_mid_center
        err_x = frame_center[0] - cx
        err_y = frame_center[1] - cy

        if not tracking_active:
            status_text = 'WAIT'
            if send_due():
                serial_dev.write_str('wait\n')
                tracking_active = True
                open_sent = False
        elif not open_sent:
            if abs(err_x) <= ALIGN_THR_X and abs(err_y) <= ALIGN_THR_Y:
                status_text = 'OPEN'
                if send_due():
                    serial_dev.write_str('ok\n')
                    open_sent = True
                    led.value(1)
            else:
                status_text = f'TRACK dx={err_x:.1f} dy={err_y:.1f}'
                led.value(0)
                if send_due():
                    serial_dev.write_str(f'x:{err_x:.2f},y:{err_y:.2f}\n')
        else:
            status_text = 'OPEN'
    else:
        lost_map_cnt += 1
        if lost_map_cnt > MAX_MAP_LOST:
            tracking_active = False
            open_sent = False
            status_text = 'MAP LOST'
            last_mid_center = None
            last_mid_rect = None
            led.value(0)
        else:
            status_text = 'MAP SEARCHING'

    if not show:
        return

    if outer_rect is not None and inner_rect is not None and mid_rect is not None and last_mid_center is not None:
        cx, cy = last_mid_center
        cv2.circle(frame_bgr, (cx, cy), 4, (0, 255, 255), -1)
        cv2.circle(frame_bgr, frame_center, 4, (255, 255, 0), -1)
        cv2.line(frame_bgr, frame_center, (cx, cy), (255, 0, 0), 2)
        cv2.putText(
            frame_bgr, f'RECT ({cx}, {cy})', (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
        )
        draw_quad(frame_bgr, outer_rect, (0, 255, 0), 2)
        draw_quad(frame_bgr, inner_rect, (255, 0, 0), 2)
        draw_quad(frame_bgr, mid_rect, (0, 0, 255), 2)

    cv2.putText(
        frame_bgr, status_text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
    )
    cv2.putText(
        frame_bgr, f'FPS: {fps:.1f}', (w - 95, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1
    )
    cv2.putText(
        frame_bgr, f'Center=({frame_center[0]}, {frame_center[1]})', (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )

    if err_x is not None and err_y is not None:
        cv2.putText(
            frame_bgr, f'Err=({err_x:.1f}, {err_y:.1f})', (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    display_img = image.cv2image(frame_bgr, bgr=True)
    disp.show(display_img)