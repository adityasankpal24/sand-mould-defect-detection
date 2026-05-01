# ============================================================
#   SAND MOULD DEFECT DETECTION — v9 FINAL
#   Run: python main_pipeline_v9_final.py
#
#   CHANGES FROM v9:
#     - Greyscale diff detection runs ONLY inside mould contour
#     - CLAHE local contrast enhancement inside mould only
#     - Hole detection uses dark-region analysis (no reference needed)
#     - Dual mode: reference-based diff + standalone dark-hole scan
#     - Chessboard ROI locked on first detection (never recalculates)
# ============================================================

import cv2
import numpy as np 
import glob
import os
import sys
import time
from datetime import datetime

# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────
LEFT_URL  = "rtsp://admin:Admin%40123@192.168.1.102:554/cam/realmonitor?channel=1&subtype=1"
RIGHT_URL = "rtsp://admin:Admin%40123@192.168.1.101:554/cam/realmonitor?channel=1&subtype=1"

CHECKERBOARD   = (9, 6)
SQUARE_SIZE_MM = 20
TOTAL_CALIB    = 15
TOTAL_MOULD    = 20
FRAME_SIZE     = (640, 480)

CAL_DIR    = "calibration_images"
MOULD_DIR  = "reference_mould"
CALIB_FILE = "stereo_calibration.npz"

# Sandy brown/tan mould colour (pre-tuned from your image)
LOWER_MOULD = np.array([8,  60,  80],  dtype=np.uint8)
UPPER_MOULD = np.array([25, 200, 220], dtype=np.uint8)

# Shadow / highlight exclusion
SHADOW_V_MIN    = 40
HIGHLIGHT_V_MAX = 240

# Mould size filter
MIN_MOULD_AREA = 3000

# ── HOLE DETECTION (standalone — no reference needed) ──
# A hole appears as a significantly darker region INSIDE the mould
# We compare each pixel's brightness to the LOCAL AVERAGE of the mould
HOLE_DARKNESS_RATIO  = 0.68   # pixel must be this fraction of mean brightness
                               # lower = more sensitive  (try 0.60–0.75)
HOLE_MIN_AREA        = 120    # px² minimum hole size
HOLE_MAX_AREA        = 8000   # px² ignore huge dark areas (shadows on edge)
HOLE_CIRCULARITY_MIN = 0.20   # holes are roughly circular

# ── REFERENCE-BASED detection (when R is pressed) ──
DEFAULT_SENSITIVITY  = 25
MIN_CONTOUR_AREA     = 180

# Chessboard ROI expansion
BOARD_ROI_EXPAND    = 0.6
BOARD_ROI_SIDE_PAD  = 0.15


# ──────────────────────────────────────────────
#  ZOOM CONTROLLER
# ──────────────────────────────────────────────
class ZoomView:
    def __init__(self):
        self.zoom=1.0; self.cx=0.5; self.cy=0.5
        self.step=0.25; self.pan=0.05; self.max_z=8.0

    def apply(self, frame):
        h,w = frame.shape[:2]
        cw=int(w/self.zoom); ch=int(h/self.zoom)
        x1=max(0,min(int(self.cx*w)-cw//2, w-cw))
        y1=max(0,min(int(self.cy*h)-ch//2, h-ch))
        return cv2.resize(frame[y1:y1+ch, x1:x1+cw], (w,h))

    def handle_key(self, key):
        ch = chr(key & 0xFF) if key != -1 else ''
        if   ch in ('+','='): self.zoom=min(self.zoom+self.step,self.max_z); return True
        elif ch == '-':       self.zoom=max(self.zoom-self.step,1.0);        return True
        elif ch in ('r','R'): self.zoom=1.0; self.cx=0.5; self.cy=0.5;      return True
        elif ch in ('w','W'): self.cy=max(0.0,self.cy-self.pan/self.zoom);   return True
        elif ch in ('s','S'): self.cy=min(1.0,self.cy+self.pan/self.zoom);   return True
        elif ch in ('a','A'): self.cx=max(0.0,self.cx-self.pan/self.zoom);   return True
        elif ch in ('d','D'): self.cx=min(1.0,self.cx+self.pan/self.zoom);   return True
        return False

    def hud(self, frame, extra=""):
        cv2.putText(frame,
                    f"Zoom {self.zoom:.1f}x  +/-  WASD=pan  R=reset  {extra}",
                    (10, frame.shape[0]-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)


# ──────────────────────────────────────────────
#  AUTO MOULD DETECTOR
# ──────────────────────────────────────────────
class AutoMouldDetector:
    def __init__(self):
        self.board_roi    = None   # locked after first detection
        self.board_locked = False  # True once ROI is locked
        self.last_mask    = None
        self.last_contour = None

        self.k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
        self.k_dilate= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

    def try_lock_board(self, frame_bgr):
        """
        Attempt to find chessboard and lock the ROI.
        Once locked, never recalculates (board is fixed).
        Returns True if board was found (or already locked).
        """
        if self.board_locked:
            return True

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD, cv2.CALIB_CB_FAST_CHECK)

        if not ret:
            return False

        corners = cv2.cornerSubPix(
            gray, corners, (11,11), (-1,-1),
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        xs = corners[:,0,0]; ys = corners[:,0,1]
        bx,by = int(xs.min()), int(ys.min())
        bw = int(xs.max()-xs.min())
        bh = int(ys.max()-ys.min())

        H,W = frame_bgr.shape[:2]
        pad_x = int(bw * BOARD_ROI_SIDE_PAD)
        pad_y = int(bh * BOARD_ROI_EXPAND)

        rx = max(0,      bx - pad_x)
        ry = max(0,      by - pad_y)
        rw = min(W-rx,   bw + pad_x*2)
        rh = min(H-ry,   bh + pad_y + bh)

        self.board_roi    = (rx, ry, rw, rh)
        self.board_locked = True
        print(f"  ✅  Board ROI locked: {self.board_roi}")
        return True

    def get_mould_mask(self, frame_bgr):
        """
        Returns (mould_mask, mould_contour, roi_rect)
        Uses locked ROI — searches only inside chessboard region.
        """
        H,W = frame_bgr.shape[:2]

        if self.board_roi is not None:
            rx,ry,rw,rh = self.board_roi
        else:
            rx,ry,rw,rh = 0,0,W,H

        roi = frame_bgr[ry:ry+rh, rx:rx+rw]

        # Color filter
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        cmask = cv2.inRange(hsv, LOWER_MOULD, UPPER_MOULD)

        # Shadow / highlight filter
        v = hsv[:,:,2]
        cmask = cv2.bitwise_and(cmask, (v > SHADOW_V_MIN).astype(np.uint8)*255)
        cmask = cv2.bitwise_and(cmask, (v < HIGHLIGHT_V_MAX).astype(np.uint8)*255)

        # Texture filter — mould has texture, walls don't
        gray_roi  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        lap       = cv2.Laplacian(gray_roi, cv2.CV_64F)
        texture   = (np.abs(lap) > 3).astype(np.uint8)*255
        texture   = cv2.dilate(texture, self.k_dilate, iterations=2)
        cmask     = cv2.bitwise_and(cmask, texture)

        # Morphology cleanup
        cmask = cv2.morphologyEx(cmask, cv2.MORPH_OPEN,  self.k_open,  iterations=1)
        cmask = cv2.morphologyEx(cmask, cv2.MORPH_CLOSE, self.k_close, iterations=2)

        # Largest blob = mould
        cnts,_ = cv2.findContours(cmask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        valid  = [c for c in cnts if cv2.contourArea(c) > MIN_MOULD_AREA]
        if not valid:
            return None, None, (rx,ry,rw,rh)

        best = max(valid, key=cv2.contourArea)

        # Translate to full-frame coordinates
        full_cnt = best.copy()
        full_cnt[:,:,0] += rx
        full_cnt[:,:,1] += ry

        full_mask = np.zeros((H,W), dtype=np.uint8)
        cv2.drawContours(full_mask, [full_cnt], -1, 255, -1)

        self.last_mask    = full_mask
        self.last_contour = full_cnt
        return full_mask, full_cnt, (rx,ry,rw,rh)

    # ── HOLE DETECTION — standalone, no reference needed ──────
    def detect_holes(self, frame_bgr, mould_mask, mould_contour):
        """
        Finds dark holes inside the mould using LOCAL brightness comparison.
        Works WITHOUT a reference frame.

        Strategy:
          1. Extract mould pixels only (masked)
          2. Apply CLAHE to enhance local contrast
          3. Compute mean brightness of mould surface
          4. Threshold pixels significantly darker than mean → holes
          5. Filter by area, circularity, position (must be inside contour)

        Returns (hole_found, hole_mask, annotated_display_patch,
                 list_of_hole_contours)
        """
        if mould_mask is None or mould_contour is None:
            return False, None, None, []

        H,W = frame_bgr.shape[:2]

        # Get tight bounding rect of mould
        bx,by,bw,bh = cv2.boundingRect(mould_contour)
        bx = max(0,bx); by = max(0,by)
        bw = min(W-bx,bw); bh = min(H-by,bh)
        if bw < 10 or bh < 10:
            return False, None, None, []

        # Extract ROI
        roi_bgr  = frame_bgr[by:by+bh, bx:bx+bw].copy()
        roi_mask = mould_mask[by:by+bh, bx:bx+bw]

        # Convert to greyscale — ONLY process inside mould
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # ── CLAHE: enhance local contrast inside mould only ──
        # Apply CLAHE to full ROI but we only care about masked pixels
        gray_eq = self.clahe.apply(gray)

        # Zero out pixels outside mould mask (so they don't affect stats)
        gray_masked = cv2.bitwise_and(gray_eq, gray_eq, mask=roi_mask)

        # ── Compute mean brightness of mould surface only ──
        mould_pixels = gray_eq[roi_mask > 0]
        if len(mould_pixels) == 0:
            return False, None, None, []

        mean_val = float(np.mean(mould_pixels))
        std_val  = float(np.std(mould_pixels))

        # Threshold: pixels darker than (mean - 1.5*std) are hole candidates
        # This adapts automatically to the lighting conditions
        dark_threshold = max(0, mean_val - 1.5 * std_val)
        dark_threshold = min(dark_threshold, mean_val * HOLE_DARKNESS_RATIO)

        dark_mask = np.zeros_like(gray_eq)
        dark_mask[gray_eq < dark_threshold] = 255

        # Must be inside mould
        dark_mask = cv2.bitwise_and(dark_mask, dark_mask, mask=roi_mask)

        # Morphology: remove speckle noise, keep solid dark blobs
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN,  k3, iterations=1)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, k5, iterations=1)

        # Find hole contours
        cnts,_ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

        hole_cnts_full = []   # in full-frame coords
        hole_found     = False

        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < HOLE_MIN_AREA or area > HOLE_MAX_AREA:
                continue

            perim = cv2.arcLength(cnt, True)
            if perim == 0: continue
            circ = 4*np.pi*area / (perim*perim + 1e-6)
            if circ < HOLE_CIRCULARITY_MIN:
                continue

            # Must be fully inside mould contour
            # (check centroid is inside)
            M = cv2.moments(cnt)
            if M['m00'] == 0: continue
            cx_ = int(M['m10']/M['m00']) + bx
            cy_ = int(M['m01']/M['m00']) + by
            if cv2.pointPolygonTest(mould_contour,
                                    (float(cx_), float(cy_)), False) < 0:
                continue

            hole_found = True
            # Translate to full-frame coords
            fc = cnt.copy()
            fc[:,:,0] += bx; fc[:,:,1] += by
            hole_cnts_full.append((fc, area, circ))

        # Build full-frame hole mask
        hole_mask = np.zeros((H,W), dtype=np.uint8)
        for fc,_,_ in hole_cnts_full:
            cv2.drawContours(hole_mask, [fc], -1, 255, -1)

        return hole_found, hole_mask, dark_mask, hole_cnts_full


# ──────────────────────────────────────────────
#  REFERENCE-BASED DEFECT DETECTOR (backup/extra)
# ──────────────────────────────────────────────
class RefDetector:
    def __init__(self):
        self.ref_gray  = None
        self.ref_frame = None
        self.sensitivity    = DEFAULT_SENSITIVITY
        self.min_cnt_area   = MIN_CONTOUR_AREA
        self.use_adaptive   = True
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

    def detect(self, frame_bgr, display_frame, mould_mask=None):
        if self.ref_gray is None:
            return False, None, None

        # Greyscale + CLAHE inside mould only
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        gray = cv2.GaussianBlur(gray,(5,5),1.0)

        if mould_mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=mould_mask)

        diff = cv2.absdiff(self.ref_gray, gray).astype(np.uint8)

        if self.use_adaptive:
            try:
                thresh = cv2.adaptiveThreshold(
                    diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2)
            except:
                _, thresh = cv2.threshold(
                    diff, self.sensitivity, 255, cv2.THRESH_BINARY)
        else:
            _, thresh = cv2.threshold(
                diff, self.sensitivity, 255, cv2.THRESH_BINARY)

        if mould_mask is not None:
            thresh = cv2.bitwise_and(thresh, thresh, mask=mould_mask)

        k = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k, iterations=1)

        cnts,_ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found = False
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < self.min_cnt_area: continue
            p = cv2.arcLength(cnt,True)
            if p == 0: continue
            if 4*np.pi*area/(p*p+1e-6) > 0.85 and area < 400: continue
            found = True
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(display_frame,(x,y),(x+w,y+h),(255,0,255),2)
            cv2.putText(display_frame,f"RefDef:{area:.0f}px",
                        (x,max(0,y-5)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,0,255),1)
        return found, thresh, diff

    def set_reference(self, frame_bgr, mould_mask=None):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        gray = cv2.GaussianBlur(gray,(5,5),1.0)
        if mould_mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=mould_mask)
        self.ref_gray  = gray
        self.ref_frame = frame_bgr.copy()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"reference_{ts}.png", frame_bgr)
        print(f"  ✅  Reference saved → reference_{ts}.png")
        return ts

    def clear(self):
        self.ref_gray = self.ref_frame = None


# ──────────────────────────────────────────────
#  UTILITIES
# ──────────────────────────────────────────────
def open_cameras():
    capL = cv2.VideoCapture(LEFT_URL,  cv2.CAP_FFMPEG)
    capR = cv2.VideoCapture(RIGHT_URL, cv2.CAP_FFMPEG)
    capL.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capR.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capL.isOpened(): print("❌  Left camera failed");  sys.exit(1)
    if not capR.isOpened(): print("❌  Right camera failed"); sys.exit(1)
    print("✅  Both cameras connected")
    return capL, capR

def read_both(capL, capR):
    for _ in range(2): capL.grab(); capR.grab()
    retL, fL = capL.retrieve()
    retR, fR = capR.retrieve()
    return retL and retR, fL, fR

def resize(f): return cv2.resize(f, FRAME_SIZE)
def banner(t): print(f"\n{'─'*60}\n  {t}\n{'─'*60}")

def safe_destroy(name):
    try:
        if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 0:
            cv2.destroyWindow(name)
    except: pass


# ──────────────────────────────────────────────
#  STEP 6 — LIVE DETECTION
# ──────────────────────────────────────────────
def step6_live_detection():
    banner("STEP 6 — Live Detection  v9 Final")
    print("  System starts detecting holes IMMEDIATELY — no reference needed.")
    print("  R = capture reference (enables extra ref-based detection)")
    print("  C = clear reference")
    print("  D = toggle debug windows")
    print("  T = toggle adaptive threshold")
    print("  1/2 = sensitivity -/+   3/4 = min area -/+")
    print("  5/6 = hole darkness threshold tighter/looser")
    print("  +/- WASD R = zoom/pan/reset")
    print("  ESC = quit\n")

    # Load calibration
    maps = None
    if os.path.exists(CALIB_FILE):
        data = np.load(CALIB_FILE)
        h,w  = FRAME_SIZE[1], FRAME_SIZE[0]
        def make_maps(mtx,dist):
            nm,_ = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1)
            return cv2.initUndistortRectifyMap(mtx,dist,None,nm,(w,h),5)
        m1L,m2L = make_maps(data["mtxL"],data["distL"])
        m1R,m2R = make_maps(data["mtxR"],data["distR"])
        maps = (m1L,m2L,m1R,m2R)
        print("  ✅  Undistortion active")

    capL, capR = open_cameras()

    amdL = AutoMouldDetector()
    amdR = AutoMouldDetector()
    refL = RefDetector()
    refR = RefDetector()
    zL,  zR  = ZoomView(), ZoomView()

    show_debug = False
    hole_thresh = HOLE_DARKNESS_RATIO   # live-tunable
    fps = fc = 0
    t_fps = t_alert = time.time()

    while True:
        ok, fL, fR = read_both(capL, capR)
        if not ok: continue

        if maps:
            fL = cv2.remap(fL,maps[0],maps[1],cv2.INTER_LINEAR)
            fR = cv2.remap(fR,maps[2],maps[3],cv2.INTER_LINEAR)
        fL = resize(fL); fR = resize(fR)

        fc += 1; now = time.time()
        if now - t_fps >= 1.0:
            fps = fc; fc = 0; t_fps = now

        # ── Try to lock chessboard ROI (only runs until locked) ──
        amdL.try_lock_board(fL)
        amdR.try_lock_board(fR)

        # ── Get mould mask ──
        maskL, cntL, roiL = amdL.get_mould_mask(fL)
        maskR, cntR, roiR = amdR.get_mould_mask(fR)

        # ── Hole detection (standalone — no reference) ──
        holeL, hole_maskL, dark_maskL, hole_cnts_L = \
            amdL.detect_holes(fL, maskL, cntL)
        holeR, hole_maskR, dark_maskR, hole_cnts_R = \
            amdR.detect_holes(fR, maskR, cntR)

        # ── Reference-based detection (extra layer when ref is set) ──
        refDefL = refDefR = False
        if refL.ref_gray is not None:
            refDefL, _, _ = refL.detect(fL, fL.copy(), mould_mask=maskL)
        if refR.ref_gray is not None:
            refDefR, _, _ = refR.detect(fR, fR.copy(), mould_mask=maskR)

        # ── Build display frames ──
        dispL = fL.copy(); dispR = fR.copy()

        # Draw mould outline
        if cntL is not None:
            cv2.drawContours(dispL, [cntL], -1, (0,255,100), 2)
            bx,by,bw,bh = cv2.boundingRect(cntL)
            cv2.rectangle(dispL,(bx,by),(bx+bw,by+bh),(0,255,255),1)

        if cntR is not None:
            cv2.drawContours(dispR, [cntR], -1, (0,255,100), 2)
            bx,by,bw,bh = cv2.boundingRect(cntR)
            cv2.rectangle(dispR,(bx,by),(bx+bw,by+bh),(0,255,255),1)

        # Draw detected holes
        for fc_info, disp in [(hole_cnts_L, dispL), (hole_cnts_R, dispR)]:
            for (hcnt, area, circ) in fc_info:
                hx,hy,hw,hh = cv2.boundingRect(hcnt)
                cv2.rectangle(disp,(hx,hy),(hx+hw,hy+hh),(0,0,255),2)
                cv2.drawContours(disp,[hcnt],-1,(0,128,255),-1)  # filled orange
                cv2.putText(disp,
                            f"HOLE {area:.0f}px c:{circ:.2f}",
                            (hx, max(hy-6,12)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1)

        # ── Overall status ──
        is_defect = holeL or holeR or refDefL or refDefR
        mould_ok  = (cntL is not None) or (cntR is not None)

        if not mould_ok:
            label = "NO MOULD FOUND"; color = (0,165,255)
        elif is_defect:
            label = "DEFECT DETECTED"; color = (0,0,255)
        else:
            label = "MOULD OK"; color = (0,255,0)

        board_L = "ROI:locked" if amdL.board_locked else "ROI:searching…"
        board_R = "ROI:locked" if amdR.board_locked else "ROI:searching…"

        for disp, bt in [(dispL, board_L), (dispR, board_R)]:
            cv2.putText(disp, label, (20,45),
                        cv2.FONT_HERSHEY_SIMPLEX,1.4,color,3)
            cv2.putText(disp,
                        f"{bt}  Ref:{'Y' if refL.ref_gray is not None else 'N (standalone mode)'}",
                        (10,72), cv2.FONT_HERSHEY_SIMPLEX,0.5,(180,180,180),1)
            cv2.putText(disp,
                        f"Holes L:{len(hole_cnts_L)}  R:{len(hole_cnts_R)}  "
                        f"DarkThresh:{hole_thresh:.2f}  FPS:{fps}",
                        (10,90), cv2.FONT_HERSHEY_SIMPLEX,0.5,(180,180,180),1)

        # Flash red border
        if is_defect and int(now*3)%2 == 0:
            for d in (dispL, dispR):
                cv2.rectangle(d,(0,0),(FRAME_SIZE[0]-1,FRAME_SIZE[1]-1),(0,0,255),4)

        if is_defect and now-t_alert > 3:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  DEFECT DETECTED!")
            t_alert = now

        # Debug windows — CLAHE greyscale inside mould only
        if show_debug:
            for name, frame_bgr, mask in [
                    ("CLAHE Mould L", fL, maskL),
                    ("CLAHE Mould R", fR, maskR)]:
                if mask is not None:
                    gray_eq = amdL.clahe.apply(
                        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
                    vis = cv2.bitwise_and(gray_eq, gray_eq, mask=mask)
                    cv2.imshow(name, vis)
            if dark_maskL is not None:
                cv2.imshow("Dark Mask L", dark_maskL)
            if dark_maskR is not None:
                cv2.imshow("Dark Mask R", dark_maskR)
        else:
            for n in ("CLAHE Mould L","CLAHE Mould R",
                      "Dark Mask L","Dark Mask R"):
                safe_destroy(n)

        vL = zL.apply(dispL); zL.hud(vL)
        vR = zR.apply(dispR); zR.hud(vR)
        cv2.imshow("Left Camera",  vL)
        cv2.imshow("Right Camera", vR)

        key = cv2.waitKey(1)
        zL.handle_key(key); zR.handle_key(key)
        k = key & 0xFF

        if k == 27: break
        elif k == ord('r'):
            refL.set_reference(fL, mould_mask=maskL)
            refR.set_reference(fR, mould_mask=maskR)
        elif k == ord('c'):
            refL.clear(); refR.clear()
            print("  Reference cleared — running in standalone mode")
        elif k == ord('d'):
            show_debug = not show_debug
            print(f"  Debug: {'ON' if show_debug else 'OFF'}")
        elif k == ord('t'):
            refL.use_adaptive = refR.use_adaptive = not refL.use_adaptive
            print(f"  Adaptive threshold: {'ON' if refL.use_adaptive else 'OFF'}")
        elif k == ord('1'):
            refL.sensitivity = refR.sensitivity = max(10, refL.sensitivity-5)
            print(f"  Ref sensitivity: {refL.sensitivity}")
        elif k == ord('2'):
            refL.sensitivity = refR.sensitivity = min(100, refL.sensitivity+5)
            print(f"  Ref sensitivity: {refL.sensitivity}")
        elif k == ord('3'):
            refL.min_cnt_area = refR.min_cnt_area = \
                max(50, refL.min_cnt_area-50)
            print(f"  Min area: {refL.min_cnt_area}")
        elif k == ord('4'):
            refL.min_cnt_area = refR.min_cnt_area = \
                min(2000, refL.min_cnt_area+50)
            print(f"  Min area: {refL.min_cnt_area}")
        elif k == ord('5'):
            # Make hole detection MORE strict (less sensitive)
            hole_thresh = min(0.90, hole_thresh + 0.02)
            amdL.clahe = amdR.clahe = cv2.createCLAHE(
                clipLimit=3.0, tileGridSize=(8,8))
            print(f"  Hole darkness ratio: {hole_thresh:.2f}  (stricter)")
        elif k == ord('6'):
            # Make hole detection MORE sensitive
            hole_thresh = max(0.40, hole_thresh - 0.02)
            print(f"  Hole darkness ratio: {hole_thresh:.2f}  (more sensitive)")

    capL.release(); capR.release()
    cv2.destroyAllWindows()
    print("✅  Detection stopped")


# ──────────────────────────────────────────────
#  STEP 7 — COLOUR TUNER
# ──────────────────────────────────────────────
def step7_color_tuner():
    banner("STEP 7 — Colour Tuner")
    print("  Adjust until ONLY the mould is visible → ESC to print values\n")

    capL = cv2.VideoCapture(LEFT_URL, cv2.CAP_FFMPEG)
    capL.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    z = ZoomView()

    cv2.namedWindow("Color Tuner")
    cv2.createTrackbar("H low",  "Color Tuner", int(LOWER_MOULD[0]),179,lambda x:None)
    cv2.createTrackbar("H high", "Color Tuner", int(UPPER_MOULD[0]),179,lambda x:None)
    cv2.createTrackbar("S low",  "Color Tuner", int(LOWER_MOULD[1]),255,lambda x:None)
    cv2.createTrackbar("S high", "Color Tuner", int(UPPER_MOULD[1]),255,lambda x:None)
    cv2.createTrackbar("V low",  "Color Tuner", int(LOWER_MOULD[2]),255,lambda x:None)
    cv2.createTrackbar("V high", "Color Tuner", int(UPPER_MOULD[2]),255,lambda x:None)

    while True:
        for _ in range(2): capL.grab()
        ret, frame = capL.retrieve()
        if not ret: continue
        frame = resize(frame)

        hl=cv2.getTrackbarPos("H low", "Color Tuner")
        hh=cv2.getTrackbarPos("H high","Color Tuner")
        sl=cv2.getTrackbarPos("S low", "Color Tuner")
        sh=cv2.getTrackbarPos("S high","Color Tuner")
        vl=cv2.getTrackbarPos("V low", "Color Tuner")
        vh=cv2.getTrackbarPos("V high","Color Tuner")

        lower = np.array([hl,sl,vl],dtype=np.uint8)
        upper = np.array([hh,sh,vh],dtype=np.uint8)
        mask  = cv2.inRange(cv2.cvtColor(frame,cv2.COLOR_BGR2HSV),lower,upper)
        viz   = cv2.bitwise_and(frame,frame,mask=mask)
        cv2.putText(viz,f"({hl},{sl},{vl}) → ({hh},{sh},{vh})",
                    (5,22),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),1)

        cv2.imshow("Color Tuner", z.apply(viz))
        cv2.imshow("Original",    z.apply(frame))
        key = cv2.waitKey(1)
        if z.handle_key(key): continue
        if key & 0xFF == 27:
            print(f"\n  LOWER_MOULD = np.array([{hl}, {sl}, {vl}])")
            print(f"  UPPER_MOULD = np.array([{hh}, {sh}, {vh}])")
            break

    capL.release(); cv2.destroyAllWindows()


# ──────────────────────────────────────────────
#  STEPS 1–5  (calibration)
# ──────────────────────────────────────────────
def step1_capture_calibration():
    banner("STEP 1 — Capture calibration pairs")
    os.makedirs(f"{CAL_DIR}/left",  exist_ok=True)
    os.makedirs(f"{CAL_DIR}/right", exist_ok=True)
    for f in glob.glob(f"{CAL_DIR}/left/*.png")+glob.glob(f"{CAL_DIR}/right/*.png"):
        os.remove(f)
    capL, capR = open_cameras()
    zL, zR = ZoomView(), ZoomView(); count = 0
    while count < TOTAL_CALIB:
        ok, fL, fR = read_both(capL, capR)
        if not ok: continue
        fL, fR = resize(fL), resize(fR)
        vL, vR = zL.apply(fL), zR.apply(fR)
        zL.hud(vL,"SPACE=capture"); zR.hud(vR)
        combo = cv2.hconcat([vL,vR])
        cv2.putText(combo,f"Pairs:{count}/{TOTAL_CALIB}",
                    (10,28),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        cv2.imshow("Step 1",combo)
        key = cv2.waitKey(1)
        zL.handle_key(key); zR.handle_key(key)
        if key & 0xFF == 32:
            cv2.imwrite(f"{CAL_DIR}/left/{count}.png",  fL)
            cv2.imwrite(f"{CAL_DIR}/right/{count}.png", fR)
            count += 1; print(f"  Pair {count}")
        elif key & 0xFF == 27: break
    capL.release(); capR.release(); cv2.destroyAllWindows()

def step2_validate_chessboard():
    banner("STEP 2 — Validate chessboard")
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001)
    images=sorted(glob.glob(f"{CAL_DIR}/left/*.png"))
    z=ZoomView(); valid=0; invalid=[]
    for path in images:
        frame=cv2.imread(path)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        ret,corners=cv2.findChessboardCorners(gray,CHECKERBOARD,None)
        if ret:
            cv2.drawChessboardCorners(frame,CHECKERBOARD,corners,ret)
            cv2.putText(frame,"VALID",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
            valid += 1
        else:
            cv2.putText(frame,"INVALID",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)
            invalid.append(path)
        while True:
            view=z.apply(frame.copy()); z.hud(view,"any key=next")
            cv2.imshow("Step 2",view)
            key=cv2.waitKey(30)
            if z.handle_key(key): continue
            if key != -1: break
    cv2.destroyAllWindows()
    for bad in invalid:
        idx=os.path.splitext(os.path.basename(bad))[0]; os.remove(bad)
        r=f"{CAL_DIR}/right/{idx}.png"
        if os.path.exists(r): os.remove(r)
    print(f"✅  {valid}/{len(images)} valid")
    if len(glob.glob(f"{CAL_DIR}/left/*.png"))<6:
        print("❌  Too few — re-run Step 1"); sys.exit(1)

def step3_stereo_calibration():
    banner("STEP 3 — Stereo calibration")
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001)
    objp=np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3),np.float32)
    objp[:,:2]=np.mgrid[0:CHECKERBOARD[0],
                         0:CHECKERBOARD[1]].T.reshape(-1,2)*SQUARE_SIZE_MM
    objpoints,imgpL,imgpR=[],[],[]; gray_shape=None
    for iL,iR in zip(sorted(glob.glob(f"{CAL_DIR}/left/*.png")),
                     sorted(glob.glob(f"{CAL_DIR}/right/*.png"))):
        gL=cv2.cvtColor(cv2.imread(iL),cv2.COLOR_BGR2GRAY)
        gR=cv2.cvtColor(cv2.imread(iR),cv2.COLOR_BGR2GRAY)
        gray_shape=gL.shape
        retL,cL=cv2.findChessboardCorners(gL,CHECKERBOARD,None)
        retR,cR=cv2.findChessboardCorners(gR,CHECKERBOARD,None)
        if retL and retR:
            objpoints.append(objp)
            imgpL.append(cv2.cornerSubPix(gL,cL,(11,11),(-1,-1),criteria))
            imgpR.append(cv2.cornerSubPix(gR,cR,(11,11),(-1,-1),criteria))
    if len(objpoints)<6: print("❌  Not enough pairs"); sys.exit(1)
    retL,mtxL,distL,_,_=cv2.calibrateCamera(
        objpoints,imgpL,gray_shape[::-1],None,None)
    retR,mtxR,distR,_,_=cv2.calibrateCamera(
        objpoints,imgpR,gray_shape[::-1],None,None)
    _,_,_,_,_,R,T,_,_=cv2.stereoCalibrate(
        objpoints,imgpL,imgpR,mtxL,distL,mtxR,distR,
        gray_shape[::-1],criteria=criteria,flags=cv2.CALIB_FIX_INTRINSIC)
    np.savez(CALIB_FILE,mtxL=mtxL,distL=distL,mtxR=mtxR,distR=distR,R=R,T=T)
    print(f"✅  Saved → {CALIB_FILE}  baseline≈{np.linalg.norm(T):.1f}mm")

def step4_capture_mould():
    banner("STEP 4 — Capture mould reference")
    os.makedirs(f"{MOULD_DIR}/left",  exist_ok=True)
    os.makedirs(f"{MOULD_DIR}/right", exist_ok=True)
    capL,capR=open_cameras()
    zL,zR=ZoomView(),ZoomView(); count=0
    while count<TOTAL_MOULD:
        ok,fL,fR=read_both(capL,capR)
        if not ok: continue
        fL,fR=resize(fL),resize(fR)
        vL,vR=zL.apply(fL),zR.apply(fR)
        zL.hud(vL,"R=capture"); zR.hud(vR)
        combo=cv2.hconcat([vL,vR])
        cv2.putText(combo,f"Mould:{count}/{TOTAL_MOULD}",
                    (10,28),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        cv2.imshow("Step 4",combo)
        key=cv2.waitKey(1)
        zL.handle_key(key); zR.handle_key(key)
        if key & 0xFF == ord('r'):
            cv2.imwrite(f"{MOULD_DIR}/left/{count}.png",  fL)
            cv2.imwrite(f"{MOULD_DIR}/right/{count}.png", fR)
            count += 1
        elif key & 0xFF == 27: break
    capL.release(); capR.release(); cv2.destroyAllWindows()

def step5_stitch_reference():
    banner("STEP 5 — Stitch reference")
    for side in ("left","right"):
        paths=sorted(glob.glob(f"{MOULD_DIR}/{side}/*.png"))[::2][:10]
        images=[cv2.resize(cv2.imread(p),(320,240))
                for p in paths if cv2.imread(p) is not None]
        if len(images)<2: continue
        stitcher=cv2.Stitcher_create(cv2.Stitcher_SCANS)
        status,stitched=stitcher.stitch(images)
        out=cv2.resize(stitched,FRAME_SIZE) if status==cv2.Stitcher_OK else \
            cv2.resize(max(images,key=lambda im:cv2.Laplacian(
                cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()),FRAME_SIZE)
        cv2.imwrite(f"{MOULD_DIR}/reference_{side}.png",out)
    print("✅  Done")


# ──────────────────────────────────────────────
#  MAIN MENU
# ──────────────────────────────────────────────
STEPS = {
    "1": ("Capture calibration images",             step1_capture_calibration),
    "2": ("Validate chessboard images",             step2_validate_chessboard),
    "3": ("Stereo calibration",                     step3_stereo_calibration),
    "4": ("Capture reference mould images",         step4_capture_mould),
    "5": ("Stitch reference panorama",              step5_stitch_reference),
    "6": ("Live detection — FULLY AUTO, NO REF",    step6_live_detection),
    "7": ("Colour tuner",                           step7_color_tuner),
    "A": ("Run ALL  1→2→3→4→5→6",                  None),
}

def main():
    banner("SAND MOULD DEFECT DETECTION  v9 Final")
    for k,(name,_) in STEPS.items():
        print(f"  [{k}]  {name}")
    print()
    choice = input("Enter step: ").strip().upper()
    if choice == "A":
        step1_capture_calibration(); step2_validate_chessboard()
        step3_stereo_calibration();  step4_capture_mould()
        step5_stitch_reference();    step6_live_detection()
    elif choice in STEPS and STEPS[choice][1]:
        STEPS[choice][1]()
    else:
        print("❌  Invalid choice")

if __name__ == "__main__":
    main()