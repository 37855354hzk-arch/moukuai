from maix import image
import cv2
import time
import numpy as np
import math

from hardware import shared_gpio as led # type: ignore

LASER_SPOT_THRESH = [[93, 100, -8, 1, -3, 20]]

CAM_W, CAM_H = 320, 240
WARP_W, WARP_H = 320, 320
MAP_SEARCH_SIZE = 180
MAX_MAP_LOST = 2

MIN_OUTER_AREA = 1000
MIN_INNER_AREA = 600

MIN_LASER_AREA = 5
MAX_LASER_AREA = 150
MAX_LASER_ASPECT = 2.5

SEND_INTERVAL = 0.02

KERNEL_MAP = np.ones((3, 3), np.uint8)
LASER_LOWER = np.array([
    int(LASER_SPOT_THRESH[0][0] * 2.55),
    LASER_SPOT_THRESH[0][2] + 128,
    LASER_SPOT_THRESH[0][4] + 128,
], dtype=np.uint8)
LASER_UPPER = np.array([
    int(LASER_SPOT_THRESH[0][1] * 2.55),
    LASER_SPOT_THRESH[0][3] + 128,
    LASER_SPOT_THRESH[0][5] + 128,
], dtype=np.uint8)

last_mid_center = None
last_laser_center = None
last_M = None
last_Minv = None
lost_map_cnt = 0
next_send_time = time.monotonic()


def init_laser():
    global last_mid_center, last_laser_center, last_M, last_Minv, lost_map_cnt, next_send_time
    last_mid_center = None
    last_laser_center = None
    last_M = None
    last_Minv = None
    lost_map_cnt = 0
    next_send_time = time.monotonic()
    led.value(1)
    print('--- laser mode initialized ---')


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
        cv2.THRESH_BINARY_INV, 15, 7
    )

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, KERNEL_MAP)

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
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        child_idx = hierarchy[i][2]
        if child_idx == -1:
            continue

        child = contours[child_idx]
        child_area = cv2.contourArea(child)
        if child_area < MIN_INNER_AREA:
            continue

        child_peri = cv2.arcLength(child, True)
        child_approx = cv2.approxPolyDP(child, 0.02 * child_peri, True)
        if len(child_approx) != 4 or not cv2.isContourConvex(child_approx):
            continue

        ratio = child_area / float(area)
        if ratio < 0.05 or ratio > 0.95:
            continue

        outer = order_points(approx.reshape(4, 2).astype(np.float32))
        inner = order_points(child_approx.reshape(4, 2).astype(np.float32))
        mid = (outer + inner) / 2.0

        if best is None or area > best['area']:
            best = {'outer': outer, 'inner': inner, 'mid': mid, 'area': area}

    return best


def make_perspective(outer_rect):
    dst = np.array([
        [0, 0], [WARP_W - 1, 0],
        [WARP_W - 1, WARP_H - 1], [0, WARP_H - 1]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(outer_rect.astype(np.float32), dst)
    Minv = cv2.getPerspectiveTransform(dst, outer_rect.astype(np.float32))
    return M, Minv


def warp_point_to_src(pt, Minv):
    pts = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pts, Minv)[0][0]
    return int(mapped[0]), int(mapped[1])


def make_roi(center, size, img_w, img_h):
    if center is None:
        return None
    cx, cy = center
    half = size // 2
    x = max(0, cx - half)
    y = max(0, cy - half)
    w = min(size, img_w - x)
    h = min(size, img_h - y)
    return [int(x), int(y), int(w), int(h)]


def blob_aspect(b):
    w = max(1, b.w())
    h = max(1, b.h())
    return max(w, h) / min(w, h)


def score_laser_blob(b, ref=None):
    area = b.pixels()
    asp = blob_aspect(b)

    if area < MIN_LASER_AREA or area > MAX_LASER_AREA:
        return -1e9
    if asp > MAX_LASER_ASPECT:
        return -1e9

    cx, cy = b.cx(), b.cy()
    score = 100.0
    score -= abs(area - 30) * 0.1
    score -= abs(asp - 1.0) * 10.0

    if ref is not None:
        dx = cx - ref[0]
        dy = cy - ref[1]
        score -= math.hypot(dx, dy) * 0.15

    return score


class SimpleBlob:
    def __init__(self, cx, cy):
        self._cx = cx
        self._cy = cy

    def cx(self):
        return self._cx

    def cy(self):
        return self._cy


def detect_laser_cv2(mask, last_center=None):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid = []
    margin = 15

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_LASER_AREA or area > MAX_LASER_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        asp = max(w, max(1, h)) / min(max(1, w), max(1, h))
        if asp > MAX_LASER_ASPECT:
            continue

        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        if cx < margin or cx > WARP_W - margin or cy < margin or cy > WARP_H - margin:
            continue

        score = 100.0
        score -= abs(area - 30) * 0.1
        score -= abs(asp - 1.0) * 10.0

        if last_center is not None:
            dx = cx - last_center[0]
            dy = cy - last_center[1]
            score -= math.hypot(dx, dy) * 0.15

        valid.append({'cx': cx, 'cy': cy, 'score': score})

    if not valid:
        return None

    best = max(valid, key=lambda item: item['score'])
    return SimpleBlob(best['cx'], best['cy'])


def step_laser(frame_bgr, serial_dev, disp, show=True, fps=0.0):
    global last_mid_center, last_laser_center, last_M, last_Minv, lost_map_cnt

    h, w = frame_bgr.shape[:2]

    map_input, (off_x, off_y) = crop_around_center(frame_bgr, last_mid_center, MAP_SEARCH_SIZE)
    map_result = detect_map_rects(map_input)

    outer_rect = None
    inner_rect = None
    mid_rect = None
    M = None
    Minv = None
    map_ok = False

    if map_result is not None:
        offset = np.array([off_x, off_y], dtype=np.float32)
        outer_rect = map_result['outer'] + offset
        inner_rect = map_result['inner'] + offset
        mid_rect = map_result['mid'] + offset

        last_mid_center = (
            int(mid_rect[:, 0].mean()),
            int(mid_rect[:, 1].mean())
        )
        lost_map_cnt = 0
        map_ok = True
        M, Minv = make_perspective(outer_rect)
        last_M = M
        last_Minv = Minv
    else:
        lost_map_cnt += 1
        if last_M is not None and last_Minv is not None and lost_map_cnt <= MAX_MAP_LOST:
            map_ok = True
            M = last_M
            Minv = last_Minv
        elif lost_map_cnt > MAX_MAP_LOST:
            last_mid_center = None
            last_laser_center = None
            last_M = None
            last_Minv = None

    status_text = 'SEARCHING'
    err_x = None
    err_y = None
    payload = None
    mask_bgr = None
    laser_orig = None
    center_orig = None

    if map_ok and M is not None:
        warp_bgr = cv2.warpPerspective(frame_bgr, M, (WARP_W, WARP_H))
        lab_img = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2LAB)
        preview_mask = cv2.inRange(lab_img, LASER_LOWER, LASER_UPPER)

        if show:
            small_mask = cv2.resize(preview_mask, (80, 80), interpolation=cv2.INTER_NEAREST)
            mask_bgr = cv2.cvtColor(small_mask, cv2.COLOR_GRAY2BGR)

        laser_blob = detect_laser_cv2(preview_mask, last_laser_center)
        center_warp = (WARP_W // 2, WARP_H // 2)

        if laser_blob is not None:
            lx, ly = laser_blob.cx(), laser_blob.cy()
            last_laser_center = (lx, ly)

            err_x = center_warp[0] - lx
            err_y = ly - center_warp[1]

            laser_orig = warp_point_to_src((lx, ly), Minv)
            center_orig = warp_point_to_src(center_warp, Minv)

            payload = f'x:{err_x:.2f},y:{err_y:.2f}\n'
            status_text = f'OK dx={err_x:.1f} dy={err_y:.1f}'
        else:
            payload = 'wait\n'
            status_text = 'MAP OK, LASER LOST'
    else:
        payload = 'wait\n'
        status_text = 'MAP LOST'

    if payload is not None and send_due():
        serial_dev.write_str(payload)

    if not show:
        return

    if outer_rect is not None and inner_rect is not None and mid_rect is not None and last_mid_center is not None:
        draw_quad(frame_bgr, outer_rect, (0, 255, 0), 2)
        draw_quad(frame_bgr, inner_rect, (255, 0, 0), 2)
        draw_quad(frame_bgr, mid_rect, (0, 0, 255), 2)

        cx, cy = last_mid_center
        cv2.circle(frame_bgr, (cx, cy), 4, (0, 255, 255), -1)
        cv2.putText(
            frame_bgr, f'MAP ({cx}, {cy})', (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
        )

    if laser_orig is not None and center_orig is not None:
        cv2.circle(frame_bgr, laser_orig, 4, (0, 0, 255), -1)
        cv2.drawMarker(
            frame_bgr, laser_orig, (0, 0, 255),
            markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2
        )
        cv2.line(frame_bgr, center_orig, laser_orig, (255, 0, 0), 2)

    cv2.putText(
        frame_bgr, status_text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
    )
    cv2.putText(
        frame_bgr, f'FPS: {fps:.1f}', (w - 95, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1
    )

    if mask_bgr is None:
        mask_bgr = np.zeros((80, 80, 3), dtype=np.uint8)
    frame_bgr[h - 80:h, 0:80] = mask_bgr
    cv2.putText(
        frame_bgr, 'MASK', (5, h - 65),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1
    )

    display_img = image.cv2image(frame_bgr, bgr=True)
    disp.show(display_img)