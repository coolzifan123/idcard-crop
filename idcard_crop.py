#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
身份证照片自动裁剪 + 摆正（离线 / 纯 CPU）

处理流程
  1. 粗定位：缩到 ~480px，做彩色边缘检测，从「轮廓」和「按梯度方向过滤的霍夫直线」两路
     生成四边形候选（直线两两组合，向量化批量评估），按
     边缘支撑度 × 长宽比(85.6:54) × 内侧边距均匀度 × 对比强度 × 面积 打分取最优。
     边缘支撑度要求卡片内外两侧的色差方向沿整条边保持一致，纹理/文字区域不会得高分。
  2. 精修：回到原图分辨率，沿四条边逐点找边缘（优先"内侧颜色像卡片边距"的那条），
     先宽后窄两轮 RANSAC 拟合直线后求交点。
     身份证是圆角，用直线交点得到"虚拟直角"，透视变换后四边严格对齐。
  3. 透视矫正：长边映射为宽，输出标准比例横版图（默认 1011x638 = 300dpi 实物尺寸）。
     到这一步只剩 0° / 180° 两种可能。
  4. 方向判断：
       人像面 —— YuNet 人脸检测（0.2MB ONNX），人脸在右侧且五官正立 → 正向；
       国徽面 —— 国徽（红色圆形色块）应在左上角；
       都判断不了时（如头像被反光盖住），可选用 RapidOCR 兜底（装了才用）：只认身份证专属字样
       （居民身份证 / 公民身份号码…），出现行驶证、驾驶证等其他证件字样时返回 unknown，不会被当成国徽面。
  5. 质量检查：清晰度、反光，供前端提示"请重拍"。
  6. 证件核验（可选，verify=True）：对每张卡跑一次 OCR，用"专属字样 + 排除字样"确认它确实是身份证的
     那一面，结论分 verified / rejected / uncertain 三种。输入里可能混有其他证件时必须打开。

依赖：Python>=3.8、numpy、opencv-python 或 opencv-python-headless（>=4.8，5.x 亦可）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Tuple

import cv2
import numpy as np

CARD_W_MM, CARD_H_MM, CORNER_R_MM = 85.6, 54.0, 3.18  # ISO/IEC 7810 ID-1
CARD_RATIO = CARD_W_MM / CARD_H_MM                      # ≈ 1.585

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FACE_MODEL = os.path.join(HERE, "models", "face_detection_yunet_2023mar.onnx")

WORK_SIZE = 480                            # 粗定位时的长边像素
LAB_WEIGHTS = np.float32([1.0, 3.0, 3.0])  # L/a/b 通道权重：放大色差，浅色卡放浅色桌面也能分开
EDGE_TAU = 7.0                             # 判定"这里有边"的最小色差（加权 Lab）


@dataclass
class IdCardResult:
    ok: bool                              # 是否成功得到卡片图
    image: Optional[np.ndarray] = None    # 裁剪 + 摆正后的 BGR 图
    side: str = "unknown"                 # front=人像面, back=国徽面, unknown
    orient_method: str = "none"           # face / emblem / ocr / none
    quad: Optional[np.ndarray] = None     # 卡片在原图中的四角（摆正后的 左上,右上,右下,左下）
    detect_score: float = 0.0             # 定位置信度 0~1
    sharpness: float = 0.0                # 清晰度（拉普拉斯方差，越大越清楚）
    glare_ratio: float = 0.0              # 最大反光块占卡面比例
    need_confirm: bool = False            # 建议前端展示结果让用户确认（定位置信度偏低 / 方向没把握 / 核验不确定）
    doc_check: str = "unchecked"          # 证件核验（verify=True 时）：verified / rejected / uncertain / unavailable
    doc_evidence: str = ""                # 核验依据，如"识别到：居民身份证"、"识别到其他证件的字样：号牌号码"
    warnings: List[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def summary(self) -> dict:
        return {
            "ok": self.ok, "need_confirm": self.need_confirm, "side": self.side, "orient_method": self.orient_method,
            "doc_check": self.doc_check, "doc_evidence": self.doc_evidence,
            "detect_score": round(self.detect_score, 3),
            "sharpness": round(self.sharpness, 1), "glare_ratio": round(self.glare_ratio, 4),
            "warnings": self.warnings, "elapsed_ms": round(self.elapsed_ms, 1),
            "quad": None if self.quad is None else np.round(self.quad, 1).tolist(),
        }


# --------------------------------------------------------------------------- 几何工具
def order_quad(pts) -> np.ndarray:
    """四个点排序为 左上、右上、右下、左下（图像坐标系顺时针）。"""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
    return np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0)


def _intersect(l1, l2) -> Optional[np.ndarray]:
    """两条直线 (vx, vy, x0, y0) 的交点。"""
    (vx1, vy1, x1, y1), (vx2, vy2, x2, y2) = l1, l2
    det = vx2 * vy1 - vx1 * vy2
    if abs(det) < 1e-6:
        return None
    t = (vx2 * (y2 - y1) - vy2 * (x2 - x1)) / det
    return np.array([x1 + t * vx1, y1 + t * vy1], np.float32)


def _sample(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """双线性采样，pts 形状 (..., 2)，返回 (..., C)。"""
    shp = pts.shape[:-1]
    flat = pts.reshape(-1, 2).astype(np.float32)
    n = len(flat)
    cols = min(n, 4096)                       # remap 要求映射表尺寸 < 32767
    rows = -(-n // cols)
    if rows * cols > n:
        flat = np.concatenate([flat, np.zeros((rows * cols - n, 2), np.float32)])
    mx = np.ascontiguousarray(flat[:, 0].reshape(rows, cols))
    my = np.ascontiguousarray(flat[:, 1].reshape(rows, cols))
    out = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(rows * cols, -1)[:n].reshape(*shp, -1)


def _geom_ok(q: np.ndarray, w: int, h: int) -> Optional[Tuple[float, float]]:
    """几何合理性：凸、面积、不出画面、内角 60°~120°。返回 (面积占比, 长宽比)。"""
    if not cv2.isContourConvex(q.reshape(-1, 1, 2)):
        return None
    frac = cv2.contourArea(q) / float(h * w)
    if frac < 0.04 or frac > 0.995:
        return None
    margin = 0.02 * max(h, w)  # 卡片被画面切掉时放弃，让前端提示重拍
    if (q[:, 0].min() < -margin or q[:, 1].min() < -margin or
            q[:, 0].max() > w - 1 + margin or q[:, 1].max() > h - 1 + margin):
        return None
    sides = [float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]
    for i in range(4):
        v1, v2 = q[i - 1] - q[i], q[(i + 1) % 4] - q[i]
        if abs(float(np.dot(v1, v2))) > 0.5 * sides[i - 1] * sides[i]:
            return None
    a, b = (sides[0] + sides[2]) / 2, (sides[1] + sides[3]) / 2
    return frac, max(a, b) / max(min(a, b), 1e-6)


# --------------------------------------------------------------------------- 1. 粗定位
def _color_gradients(lab: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """每个像素取 L/a/b 中梯度最大的通道，得到彩色梯度（int16，可直接喂给 Canny）。"""
    best_dx = best_dy = best_mag = None
    for ch, wt in zip(cv2.split(lab), LAB_WEIGHTS):
        dx = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3) * wt
        dy = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3) * wt
        mag = dx * dx + dy * dy
        if best_mag is None:
            best_dx, best_dy, best_mag = dx, dy, mag
        else:
            m = mag > best_mag
            best_dx[m], best_dy[m], best_mag[m] = dx[m], dy[m], mag[m]
    return (np.clip(best_dx, -32767, 32767).astype(np.int16),
            np.clip(best_dy, -32767, 32767).astype(np.int16))


def _quads_from_contours(edges: np.ndarray, min_area: float, max_cnt: int = 8) -> List[np.ndarray]:
    e = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    cnts = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    hulls = []
    for c in cnts:
        if len(c) < 4:
            continue
        hull = cv2.convexHull(c)
        a = cv2.contourArea(hull)
        if a >= min_area:
            hulls.append((a, hull))
    hulls.sort(key=lambda t: -t[0])
    out = []
    for _, hull in hulls[:max_cnt]:
        peri = cv2.arcLength(hull, True)
        for eps in (0.02, 0.035, 0.05):
            ap = cv2.approxPolyDP(hull, eps * peri, True)
            if len(ap) == 4:
                out.append(ap.reshape(4, 2).astype(np.float32))
                break
        out.append(cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32))
    return out


def _hough_near_horizontal(mask: np.ndarray, top_k: int) -> list:
    """mask 上找倾角 ±28° 以内的直线，按票数降序、去重，返回 [(vx, vy, x0, y0)]。"""
    h, w = mask.shape
    lines = cv2.HoughLines(mask, 1, np.pi / 180, max(20, int(0.08 * min(h, w))),
                           min_theta=np.radians(62), max_theta=np.radians(118))
    if lines is None:
        return []
    kept, ends = [], []
    for rho, theta in lines.reshape(-1, 2):
        c, s = float(np.cos(theta)), float(np.sin(theta))
        y_l, y_r = rho / s, (rho - w * c) / s          # 直线在左右边界处的 y
        if any(abs(y_l - a) < 6 and abs(y_r - b) < 6 for a, b in ends):
            continue
        ends.append((y_l, y_r))
        kept.append((-s, c, rho * c, rho * s))
        if len(kept) >= top_k:
            break
    return kept


def _segment_lines_near_horizontal(mask: np.ndarray, top_k: int) -> list:
    """HoughLinesP 找连续长线段（卡边是连续的，文字/字条断断续续），共线段合并，按 Σ长度² 排序。"""
    h, w = mask.shape
    segs = cv2.HoughLinesP(mask, 1, np.pi / 180, 15, minLineLength=int(0.08 * min(h, w)), maxLineGap=3)
    if segs is None:
        return []
    segs = segs.reshape(-1, 4).astype(np.float32)
    dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
    ok = np.abs(dy) < np.tan(np.radians(28)) * np.abs(dx)
    segs, dx, dy = segs[ok], dx[ok], dy[ok]
    if len(segs) == 0:
        return []
    length = np.hypot(dx, dy)
    slope = dy / dx
    y_l = segs[:, 1] - segs[:, 0] * slope
    y_r = y_l + w * slope
    clusters = []  # [y_l, y_r, Σlen², 代表线段]
    for i in np.argsort(-length)[:400]:
        for c in clusters:
            if abs(c[0] - y_l[i]) < 5 and abs(c[1] - y_r[i]) < 5:
                c[2] += length[i] ** 2
                break
        else:
            clusters.append([y_l[i], y_r[i], length[i] ** 2])
    clusters.sort(key=lambda c: -c[2])
    out = []
    for yl, yr, _ in clusters[:top_k]:
        d = np.array([w, yr - yl], np.float64)
        d /= np.linalg.norm(d)
        out.append((float(d[0]), float(d[1]), 0.0, float(yl)))
    return out


def _dedupe_lines(lines: list, w: int, tol: float = 5.0) -> list:
    """近水平直线按其在左右边界处的 y 去重（保留先出现的）。"""
    kept, ends = [], []
    for vx, vy, x0, y0 in lines:
        if abs(vx) < 1e-6:
            continue
        y_l = y0 - x0 * vy / vx
        y_r = y0 + (w - x0) * vy / vx
        if any(abs(y_l - a) < tol and abs(y_r - b) < tol for a, b in ends):
            continue
        ends.append((y_l, y_r))
        kept.append((vx, vy, x0, y0))
    return kept


def _line_tables(feat: np.ndarray, lines: list):
    """每条直线沿线逐像素预计算"两侧色差 ≥ 阈值"的累积和（正、负两种符号分开计），
    之后任何一段线段的边缘支撑度都能 O(1) 查出来。"""
    h, w = feat.shape[:2]
    offs = np.array([-2, -1, 0, 1, 2], np.float32)
    P0, U, T0, NN, BASE, POS, NEG = [], [], [], [], [], [], []
    off = 0
    for vx, vy, x0, y0 in lines:
        u = np.array([vx, vy], np.float32)
        u /= np.linalg.norm(u)
        nrm = np.array([-u[1], u[0]], np.float32)
        p0 = np.array([x0, y0], np.float32)
        lo, hi = -1e9, 1e9                      # 直线落在画面内的参数区间
        for p, d, m in ((p0[0], u[0], w - 1), (p0[1], u[1], h - 1)):
            if abs(d) < 1e-9:
                if p < 0 or p > m:
                    lo, hi = 0.0, 0.0
            else:
                a, b = -p / d, (m - p) / d
                lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
        t = np.arange(lo, hi, 1.0, dtype=np.float32) if hi - lo >= 4 else np.zeros(0, np.float32)
        if len(t):
            c = (p0 + t[:, None] * u)[:, None, :] + offs[None, :, None] * nrm
            d = _sample(feat, c + 2.5 * nrm) - _sample(feat, c - 2.5 * nrm)       # (N, 5, 3)
            k = np.abs(d).argmax(axis=1)
            dm = np.take_along_axis(d, k[:, None, :], axis=1)[:, 0, :]
        else:
            dm = np.zeros((0, 3), np.float32)
        POS.append(np.vstack([np.zeros((1, 3)), np.cumsum(dm >= EDGE_TAU, axis=0)]))
        NEG.append(np.vstack([np.zeros((1, 3)), np.cumsum(dm <= -EDGE_TAU, axis=0)]))
        P0.append(p0), U.append(u), T0.append(lo if len(t) else 0.0), NN.append(len(t)), BASE.append(off)
        off += len(t) + 1
    return (np.array(P0, np.float32), np.array(U, np.float32), np.array(T0, np.float32),
            np.array(NN, np.int64), np.array(BASE, np.int64),
            np.vstack(POS).astype(np.float32), np.vstack(NEG).astype(np.float32))


def _seg_support(tab, li: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """批量查询：第 li 条直线上 a→b 这段（取中间 80%）的边缘支撑度。"""
    P0, U, T0, NN, BASE, POS, NEG = tab
    p0, u = P0[li], U[li]
    ta, tb = ((a - p0) * u).sum(1), ((b - p0) * u).sum(1)
    lo, hi = np.minimum(ta, tb), np.maximum(ta, tb)
    length = hi - lo
    i0 = np.clip(np.ceil(lo + 0.1 * length - T0[li]), 0, NN[li]).astype(np.int64)
    i1 = np.clip(np.floor(hi - 0.1 * length - T0[li]) + 1, 0, NN[li]).astype(np.int64)
    i1 = np.maximum(i0, i1)
    g0, g1 = BASE[li] + i0, BASE[li] + i1
    best = np.maximum((POS[g1] - POS[g0]).max(1), (NEG[g1] - NEG[g0]).max(1))
    return np.minimum(1.0, best / np.maximum(1.0, 0.8 * length))


def _quads_from_lines(hs: list, vs: list, feat: np.ndarray, top_n: int = 40) -> List[np.ndarray]:
    """近水平线两两 × 近竖直线两两 组合成四边形，全部向量化做几何检查 + 快速打分，返回前 top_n 个。"""
    h, w = feat.shape[:2]
    nh, nv = len(hs), len(vs)
    if nh < 2 or nv < 2:
        return []
    H, V = np.array(hs, np.float64), np.array(vs, np.float64)
    vx1, vy1, x1, y1 = [H[:, i][:, None] for i in range(4)]
    vx2, vy2, x2, y2 = [V[:, i][None, :] for i in range(4)]
    det = vx2 * vy1 - vx1 * vy2
    det = np.where(np.abs(det) < 1e-6, np.nan, det)
    t = (vx2 * (y2 - y1) - vy2 * (x2 - x1)) / det
    X = np.stack([x1 + t * vx1, y1 + t * vy1], axis=-1)                  # (nh, nv, 2) 交点
    ii, jj = np.triu_indices(nh, 1)
    kk, mm = np.triu_indices(nv, 1)
    I, J = np.repeat(ii, len(kk)), np.repeat(jj, len(kk))
    K, M = np.tile(kk, len(ii)), np.tile(mm, len(ii))
    c = np.stack([X[I, K], X[I, M], X[J, M], X[J, K]], axis=1)            # (Q, 4, 2) 环形顺序
    ok = np.isfinite(c).all(axis=(1, 2))
    c = np.nan_to_num(c)
    margin = 0.02 * max(h, w)
    ok &= (c[..., 0].min(1) >= -margin) & (c[..., 1].min(1) >= -margin)
    ok &= (c[..., 0].max(1) <= w - 1 + margin) & (c[..., 1].max(1) <= h - 1 + margin)
    e = np.roll(c, -1, axis=1) - c                                        # 边向量
    cross = e[:, :, 0] * np.roll(e, -1, axis=1)[:, :, 1] - e[:, :, 1] * np.roll(e, -1, axis=1)[:, :, 0]
    ok &= (cross > 0).all(1) | (cross < 0).all(1)                         # 凸
    ln = np.linalg.norm(e, axis=2) + 1e-9
    cosang = (e * np.roll(e, -1, axis=1)).sum(2) / (ln * np.roll(ln, -1, axis=1))
    ok &= (np.abs(cosang) < 0.5).all(1)                                   # 内角 60°~120°
    nxt = np.roll(c, -1, axis=1)
    area = 0.5 * np.abs((c[:, :, 0] * nxt[:, :, 1] - nxt[:, :, 0] * c[:, :, 1]).sum(1))  # 鞋带公式
    frac = area / float(w * h)
    ok &= (frac >= 0.04) & (frac <= 0.995)
    ratio = (ln[:, 0] + ln[:, 2]) / (ln[:, 1] + ln[:, 3])
    ratio = np.maximum(ratio, 1.0 / ratio)
    dev = np.maximum(0.0, np.abs(np.log(ratio / CARD_RATIO)) - 0.09)
    s_aspect = np.exp(-(dev / 0.08) ** 2)
    ok &= s_aspect >= 0.05
    if not ok.any():
        return []
    c, I, J, K, M, frac, s_aspect = c[ok], I[ok], J[ok], K[ok], M[ok], frac[ok], s_aspect[ok]
    tab = _line_tables(feat, hs + vs)
    sup = np.stack([_seg_support(tab, I, c[:, 0], c[:, 1]), _seg_support(tab, nh + M, c[:, 1], c[:, 2]),
                    _seg_support(tab, J, c[:, 2], c[:, 3]), _seg_support(tab, nh + K, c[:, 3], c[:, 0])], axis=1)
    s_edge = np.prod(np.maximum(sup, 0.05), axis=1) ** 0.25
    quick = s_edge ** 2 * s_aspect * frac ** 0.25
    best = np.argsort(-quick)[:top_n]
    return [order_quad(c[i]) for i in best]


def _side_support(feat: np.ndarray, p0: np.ndarray, p1: np.ndarray, n: int = 40) -> Tuple[float, float, float]:
    """一条边上有多大比例的点：内侧-外侧色差 ≥ 阈值，且符号与这条边的主导符号一致。
    只看中间 80%（避开圆角）。真实卡边内外颜色差异方向一致；文字/纹理区域方向随机。
    返回 (支撑比例, 支撑点的色差中位数, 内侧边距均匀度)。"""
    v = p1 - p0
    length = float(np.hypot(v[0], v[1]))
    if length < 8:
        return 0.0, 0.0, 0.0
    u = v / length
    nin = np.array([-u[1], u[0]], np.float32)            # 顺时针排列时指向卡片内部
    t = np.linspace(0.1, 0.9, n, dtype=np.float32)
    base = p0[None, :] + t[:, None] * v[None, :]
    offs = np.array([-2, -1, 0, 1, 2], np.float32)
    c = base[:, None, :] + offs[None, :, None] * nin     # (n, 5, 2)
    d = _sample(feat, c + 2.5 * nin) - _sample(feat, c - 2.5 * nin)   # (n, 5, C) 内-外
    k = np.abs(d).argmax(axis=1)                          # 每个点取响应最强的偏移
    dm = np.take_along_axis(d, k[:, None, :], axis=1)[:, 0, :]        # (n, C)
    best, strength = 0.0, 0.0
    for ch in range(dm.shape[1]):
        x = dm[:, ch]
        strong = np.abs(x) >= EDGE_TAU
        if strong.any():
            sgn = 1.0 if x[strong].sum() >= 0 else -1.0
            good = x * sgn >= EDGE_TAU
            sup = float(np.mean(good))
            if sup > best:
                best, strength = sup, float(np.median(x[good] * sgn))
    # 身份证四周有 ≥5% 卡长的空白边距：卡边往里一点的亮度应当很均匀（文字块、头像框的内侧不是）
    inner = _sample(feat, base + max(3.0, 0.012 * length) * nin)[:, 0]
    uniform = float(np.mean(np.abs(inner - np.median(inner)) < 12.0))
    return best, strength, uniform


def _score_quad(q: np.ndarray, feat: np.ndarray) -> float:
    h, w = feat.shape[:2]
    q = order_quad(q)
    g = _geom_ok(q, w, h)
    if g is None:
        return 0.0
    frac, ratio = g
    # 长宽比：透视会让观测比例偏离 1.585，±9% 内不扣分，超出后快速衰减
    dev = max(0.0, abs(float(np.log(ratio / CARD_RATIO))) - 0.09)
    s_aspect = float(np.exp(-(dev / 0.08) ** 2))
    if s_aspect < 0.05:
        return 0.0
    # 四条边支撑度取几何平均：任何一条边不像卡边都会被重罚（内部文字行、头像边框组成的框会输）
    sides = [_side_support(feat, q[i], q[(i + 1) % 4]) for i in range(4)]
    sup = np.array([s[0] for s in sides])
    s_edge = float(np.prod(np.maximum(sup, 0.05))) ** 0.25
    # 对比强度取最弱的一条边：阴影外缘四边都比真卡边弱 → 输；头像框等内部强边不会因此加分
    s_contrast = min(1.0, min(s[1] for s in sides) / 40.0) ** 0.3
    # 内侧边距均匀度：挡掉"文字块边界"这类大框
    s_uniform = float(np.mean([s[2] for s in sides])) ** 2
    # 面积：同等边缘证据下偏好更大的框（卡片是最外层的矩形，头像、文字块只会更小）
    return s_edge ** 2 * s_aspect * s_contrast * s_uniform * frac ** 0.25


@dataclass
class _Detection:
    quad: Optional[np.ndarray]  # 原图坐标
    score: float
    scale: float                # 粗定位缩放比例
    feat: np.ndarray            # 粗定位用的加权 Lab 图


def detect_card(img: np.ndarray) -> _Detection:
    """粗定位：返回卡片四角（原图坐标，左上/右上/右下/左下）和置信度。"""
    h, w = img.shape[:2]
    s = min(1.0, WORK_SIZE / float(max(h, w)))
    small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img
    lab = cv2.cvtColor(cv2.GaussianBlur(small, (3, 3), 0), cv2.COLOR_BGR2LAB)
    feat = lab.astype(np.float32) * LAB_WEIGHTS
    dx, dy = _color_gradients(lab)
    min_area = 0.04 * small.shape[0] * small.shape[1]

    cands: List[np.ndarray] = []
    lines_h, lines_v = [], []
    gx, gy = np.abs(dx.astype(np.float32)), np.abs(dy.astype(np.float32))
    for lo, hi in ((10, 30), (20, 60), (40, 120)):
        edges = cv2.Canny(dx, dy, lo, hi, L2gradient=True)
        cands += _quads_from_contours(edges, min_area)
        if lo <= 20:
            # 边缘按梯度方向分成"近水平/近竖直"两张图（文字笔画方向杂乱，大多被滤掉），各自找直线
            on = edges > 0
            hmask = (on & (gx < 0.6 * gy)).astype(np.uint8) * 255
            vmask = np.ascontiguousarray(((on & (gy < 0.6 * gx)).astype(np.uint8) * 255).T)
            lines_h += _hough_near_horizontal(hmask, 10) + _segment_lines_near_horizontal(hmask, 8)
            lines_v += _hough_near_horizontal(vmask, 10) + _segment_lines_near_horizontal(vmask, 8)
    sh, sw = small.shape[:2]
    lines_h = _dedupe_lines(lines_h, sw)[:20]
    lines_v = [(vy, vx, y0, x0) for vx, vy, x0, y0 in _dedupe_lines(lines_v, sh)[:20]]  # 转置回原坐标
    cands += _quads_from_lines(lines_h, lines_v, feat)
    best_q, best_s, seen = None, 0.0, set()
    for q in cands:
        q = order_quad(q)
        key = tuple(np.round(q / 2).astype(int).ravel())
        if key in seen:
            continue
        seen.add(key)
        sc = _score_quad(q, feat)
        if sc > best_s:
            best_q, best_s = q, sc
    return _Detection(None if best_q is None else best_q / s, best_s, s, feat)


# --------------------------------------------------------------------------- 2. 原图分辨率精修
def _fit_line_ransac(pts: np.ndarray, wts: np.ndarray, tol: float, iters: int = 80):
    """带权 RANSAC 直线拟合，返回 (vx, vy, x0, y0) 或 None。"""
    n = len(pts)
    if n < 6:
        return None
    rng = np.random.default_rng(12345)
    idx = rng.integers(0, n, size=(iters, 2))
    p, d = pts[idx[:, 0]], pts[idx[:, 1]] - pts[idx[:, 0]]
    length = np.hypot(d[:, 0], d[:, 1])
    ok = length > 1e-3
    if not ok.any():
        return None
    p, d, length = p[ok], d[ok], length[ok]
    nrm = np.stack([-d[:, 1] / length, d[:, 0] / length], axis=1)
    diff = pts[None, :, :] - p[:, None, :]
    dist = np.abs(diff[..., 0] * nrm[:, None, 0] + diff[..., 1] * nrm[:, None, 1])
    inl = dist < tol
    best = int(np.argmax((inl * wts[None, :]).sum(axis=1)))
    sel = inl[best]
    if sel.sum() < 6:
        return None
    return tuple(cv2.fitLine(pts[sel].astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel())


def _refine_pass(img: np.ndarray, q: np.ndarray, band: int, k: int, win_sigma: float, d_in: float) -> np.ndarray:
    """沿每条边的法线方向（向内为正）找边缘点。候选边缘按 强度 × 内侧颜色像卡片的程度 × 距离权重 选取：
    卡片边距颜色取自当前边线往里 d_in 处的中位数，这样阴影外缘、背景暗带（内侧不是卡片）会被压下去。"""
    lines = []
    offs = np.arange(-band, band + 1, dtype=np.float32)
    win = np.exp(-0.5 * (offs[k:-k] / win_sigma) ** 2)   # 越靠近当前边线权重越高
    ksz = 2 * k + 1
    for i in range(4):
        p0, p1 = q[i], q[(i + 1) % 4]
        v = p1 - p0
        length = float(np.hypot(v[0], v[1]))
        u = v / length
        nrm = np.array([-u[1], u[0]], np.float32)            # 顺时针排列 → 指向卡片内部
        n = int(np.clip(length / 5, 30, 200))
        t = np.linspace(0.08, 0.92, n, dtype=np.float32)
        base = p0[None, :] + t[:, None] * v[None, :]
        grid = base[:, None, :] + offs[None, :, None] * nrm
        patch = _sample(img, grid).reshape(n, len(offs), 3).astype(np.uint8)
        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32) * LAB_WEIGHTS
        lab = cv2.GaussianBlur(lab, (ksz, 1), 0)                                  # 只沿法线方向平滑
        strength = np.abs(lab[:, 2 * k:, :] - lab[:, :-2 * k, :]).max(axis=2)    # (n, m-2k)
        # 卡片边距颜色模型
        inner = _sample(img, base + d_in * nrm).reshape(n, 1, 3).astype(np.uint8)
        inner = cv2.cvtColor(inner, cv2.COLOR_BGR2LAB).reshape(n, 3).astype(np.float32) * LAB_WEIGHTS
        card = np.median(inner, axis=0)
        sigma = max(6.0, 2.0 * float(np.linalg.norm(np.median(np.abs(inner - card), axis=0))))
        f_in = np.concatenate([lab[:, 4 * k:, :], np.repeat(lab[:, -1:, :], 2 * k, axis=1)], axis=1)  # 候选点往里 3k 的颜色（避开模糊过渡带）
        like = np.exp(-((f_in - card) ** 2).sum(axis=2) / (2 * sigma ** 2))
        idx = np.argmax(strength * like * win[None, :], axis=1)
        s = strength[np.arange(n), idx]
        pts = grid[np.arange(n), idx + k]
        keep = s >= EDGE_TAU
        line = _fit_line_ransac(pts[keep], s[keep], tol=max(1.5, 0.6 * k)) if keep.sum() >= 8 else None
        if line is None or abs(line[0] * u[0] + line[1] * u[1]) < np.cos(np.radians(6)):
            line = (u[0], u[1], p0[0], p0[1])                    # 证据不足/方向偏太多：保持原边
        lines.append(line)
    out = []
    for i in range(4):
        p = _intersect(lines[i - 1], lines[i])
        if p is None or np.linalg.norm(p - q[i]) > 2.5 * band:
            return q
        out.append(p)
    return np.array(out, np.float32)


def refine_quad(img: np.ndarray, q: np.ndarray) -> np.ndarray:
    """原图分辨率精修：沿每条边的法线方向找梯度峰值，RANSAC 拟合直线后求交点。
    先宽后窄两轮，搜索范围按卡片尺寸自适应（卡边到最近的印刷内容约有 4% 卡长的空白）。"""
    q = order_quad(q)
    long_side = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[1]),
                    np.linalg.norm(q[3] - q[2]), np.linalg.norm(q[0] - q[3]))
    k = max(1, int(round(0.003 * long_side)))       # 求导步长：随分辨率放大，适应虚焦的软边
    band1 = int(np.clip(0.04 * long_side, 8, 120))
    d_in = 0.03 * long_side                         # 往里 3% 仍在卡片空白边距内（印刷内容从 ≥5% 开始）
    q = _refine_pass(img, q, band1, k, 0.8 * band1, d_in)
    band2 = int(np.clip(0.012 * long_side, 4, 40))
    return _refine_pass(img, q, band2, k, 1.0 * band2, d_in)


# --------------------------------------------------------------------------- 3. 透视矫正
def warp_card(img: np.ndarray, q: np.ndarray, out_w: int, out_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """长边映射为宽。返回 (横版卡片图, 与输出四角对应的原图点)。"""
    q = order_quad(q)
    top = np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])
    lr = np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])
    if lr > top:                 # 卡片在照片里是竖着的：左边 → 顶边（相当于顺时针转 90°）
        q = np.roll(q, 1, axis=0)
    src, scale = img, 1.0
    long_px = max(top, lr) / 2
    if long_px > 1.5 * out_w:    # 大幅缩小时先 INTER_AREA 预缩，避免锯齿
        scale = 1.5 * out_w / long_px
        src = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    m = cv2.getPerspectiveTransform((q * scale).astype(np.float32), dst)
    card = cv2.warpPerspective(src, m, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return card, q


# --------------------------------------------------------------------------- 4. 方向判断
_tls = threading.local()  # 检测器按线程缓存：多线程服务里各线程互不干扰


def _faces(img: np.ndarray, model_path: str) -> np.ndarray:
    h, w = img.shape[:2]
    if getattr(_tls, "face_path", None) != model_path:
        _tls.face_det = cv2.FaceDetectorYN.create(model_path, "", (w, h), 0.6, 0.3, 20)
        _tls.face_path = model_path
    _tls.face_det.setInputSize((w, h))
    _, faces = _tls.face_det.detect(img)
    return np.zeros((0, 15), np.float32) if faces is None else faces


def _face_score(card: np.ndarray, model_path: str) -> float:
    """人像面正立时：人脸在右侧、眼睛在鼻子上方、鼻子在嘴上方。返回最高置信度，没有则 0。"""
    h, w = card.shape[:2]
    x_off = int(0.45 * w)
    roi = card[:, x_off:]
    best = 0.0
    for f in _faces(roi, model_path):
        x, y, fw, fh = f[:4]
        cx, cy = x + fw / 2 + x_off, y + fh / 2
        eye_y = (f[5] + f[7]) / 2
        mouth_y = (f[11] + f[13]) / 2
        upright = eye_y < f[9] < mouth_y and abs(f[7] - f[5]) < 0.5 * abs(f[6] - f[4]) + 1
        if (upright and 0.55 * w < cx < 0.97 * w and 0.1 * h < cy < 0.85 * h
                and 0.12 * h < fh < 0.75 * h):
            best = max(best, float(f[14]))
    return best


def _emblem_score(card: np.ndarray) -> float:
    """国徽面正立时国徽在左上角：找一个大小、形状都像国徽的红色色块。返回其面积占比，没有则 0。"""
    W, H = 428, 270
    small = cv2.resize(card, (W, H), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hh, ss, vv = cv2.split(hsv)
    red = (((hh <= 10) | (hh >= 165)) & (ss >= 70) & (vv >= 50)).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(red)
    best = 0.0
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        rw, rh = bw / W, bh / H
        aspect_mm = (rw * CARD_W_MM) / max(rh * CARD_H_MM, 1e-6)
        cx, cy = cen[i][0] / W, cen[i][1] / H
        if (0.10 < rw < 0.32 and 0.18 < rh < 0.52 and 0.65 < aspect_mm < 1.4
                and area > 0.35 * bw * bh and cx < 0.36 and cy < 0.5):
            best = max(best, area / float(W * H))
    return best


# --------------------------------------------------------------------------- 4b. OCR：方向兜底 + 证件核验
# 通用字样（中华人民共和国、姓名、性别、住址、出生、签发机关、有效期限）在行驶证、驾驶证、护照上也有，
# 只能当辅助证据。认定身份证只看专属字样：背面"居民身份证"，正面"公民身份号码"/"民族"。
NON_ID_KEYWORDS = (  # 其他证件特有、身份证上不会出现的字样：命中即判定"不是身份证"
    "机动车", "行驶证", "驾驶证", "号牌号码", "车辆类型", "车辆识别代号", "发动机号码", "使用性质",
    "准驾车型", "初次领证", "国籍", "护照", "PASSPORT", "居住证", "通行证", "外国人", "永久居留",
    "营业执照", "统一社会信用代码", "社会保障", "常住人口", "户主", "军官证", "士兵证", "警官证",
)
_TOP_KW = ("居民身份证", "中华人民共和国", "姓名", "性别", "民族")   # 身份证正立时在上半部的字样
_BOTTOM_KW = ("公民身份号码", "签发机关", "有效期限")                # 在下半部的字样
_DATE_RANGE = re.compile(r"\d{4}[.．]\d{1,2}[.．]\d{1,2}[-—–~至]+(?:\d{4}[.．]\d{1,2}[.．]\d{1,2}|长期)")
_ID_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)


def find_id_numbers(text: str) -> List[str]:
    """文本里校验位和出生日期都合法的 18 位公民身份号码（GB 11643）。"""
    out = []
    for m in re.finditer(r"(?<![0-9X])([1-9]\d{16}[0-9X])(?![0-9X])", text.replace("x", "X").replace("×", "X")):
        s = m.group(1)
        if "10X98765432"[sum(int(c) * w for c, w in zip(s, _ID_WEIGHTS)) % 11] != s[17]:
            continue
        try:
            date(int(s[6:10]), int(s[10:12]), int(s[12:14]))
        except ValueError:
            continue
        out.append(s)
    return out


def _lcs_len(a: str, b: str) -> int:
    dp = [0] * (len(b) + 1)
    for ca in a:
        prev = 0
        for j, cb in enumerate(b):
            cur = dp[j + 1]
            dp[j + 1] = prev + 1 if ca == cb else max(dp[j + 1], dp[j])
            prev = cur
    return dp[-1]


def _fuzzy_has(text: str, kw: str) -> bool:
    """≥5 字的关键词允许 OCR 错/漏 1 个字；短词必须原样出现（短词模糊匹配容易撞上别的词，如"发证机关"）。"""
    if kw in text:
        return True
    n = len(kw)
    return n >= 5 and any(_lcs_len(text[i:i + n + 1], kw) >= n - 1 for i in range(len(text)))


def check_document(text: str, side: str = "unknown") -> Tuple[str, str, str]:
    """根据卡面 OCR 全文核验是不是身份证。side 为已判定的面：front / back / unknown（unknown 时由文字决定）。
    返回 (结论, 面, 依据)：
      rejected  —— 出现其他证件特有的字样（行驶证、驾驶证、护照、居住证……），或号码属于港澳台居民居住证（81/82/83 开头）；
      verified  —— 出现该面的身份证专属字样：
                   背面："居民身份证"（允许错 1 字），或"签发机关"+"有效期限"+日期 同时出现（标题被反光盖住时）；
                   正面："公民身份号码"（允许错 1 字），或"民族"+校验通过的 18 位号码；
      uncertain —— 两者都没有，或字样与已判定的面矛盾，应转人工复核。"""
    neg = [k for k in NON_ID_KEYWORDS if k in text]
    nums = find_id_numbers(text)
    if any(n[:2] in ("81", "82", "83") for n in nums):
        neg.append("港澳台居民居住证号码")
    if neg:
        return "rejected", side, "识别到其他证件的字样：" + "、".join(neg)
    front_hit, back_hit = [], []
    if _fuzzy_has(text, "公民身份号码"):
        front_hit.append("公民身份号码")
    elif "民族" in text and nums:
        front_hit.append("民族+校验通过的号码")
    if _fuzzy_has(text, "居民身份证"):
        back_hit.append("居民身份证")
    elif "签发机关" in text and "有效期限" in text and _DATE_RANGE.search(text):
        back_hit.append("签发机关+有效期限+日期")
    if side not in ("front", "back"):
        if bool(front_hit) == bool(back_hit):
            return "uncertain", "unknown", "正反面的字样同时出现" if front_hit else "未识别到身份证专属字样"
        side = "front" if front_hit else "back"
    hit, other = (front_hit, back_hit) if side == "front" else (back_hit, front_hit)
    name = "正面" if side == "front" else "背面"
    if hit and not other:
        return "verified", side, "识别到：" + "、".join(hit)
    if hit:
        return "uncertain", side, "正反面的字样同时出现"
    if other:
        return "uncertain", side, "文字像身份证%s，与判定的%s不一致" % ("背面" if side == "front" else "正面", name)
    weak = "（只认出了校验通过的身份证号码）" if side == "front" and nums else ""
    return "uncertain", side, "未识别到身份证%s的专属字样%s" % (name, weak)


def _ocr_lines(card: np.ndarray) -> Optional[List[Tuple[float, str]]]:
    """RapidOCR（可选依赖，按线程缓存）识别卡面，返回 [(文字行中心 y, 文本)]；未安装或出错返回 None。"""
    eng = getattr(_tls, "ocr", None)
    if eng is False:
        return None
    try:
        if eng is None:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
            eng = _tls.ocr = RapidOCR()
        res, _ = eng(card)
    except ImportError:
        _tls.ocr = False
        return None
    except Exception:
        return None
    return [(float(np.mean([p[1] for p in box])), str(text).replace(" ", "")) for box, text, _ in (res or [])]


def _ocr_text(lines: List[Tuple[float, str]]) -> str:
    return "|".join(t for _, t in lines)


def _ocr_orientation(lines: List[Tuple[float, str]], h: int) -> Tuple[Optional[bool], str]:
    """OCR 兜底：先用 check_document 确认是身份证、是哪一面（行驶证等不会被当成国徽面），
    再看固定字样在上半部还是下半部判断正倒。返回 (是否正立 / None, 面别)。"""
    verdict, side, _ = check_document(_ocr_text(lines), "unknown")
    if verdict != "verified":
        return None, "unknown"
    top = [cy for cy, t in lines if any(k in t for k in _TOP_KW)]
    bottom = [cy for cy, t in lines if any(k in t for k in _BOTTOM_KW)]
    if top and bottom:
        return bool(np.mean(top) < np.mean(bottom)), side
    if top:
        return bool(np.mean(top) < h / 2), side
    if bottom:
        return bool(np.mean(bottom) > h / 2), side
    return None, side


def _orient(card: np.ndarray, face_model: Optional[str], use_ocr: bool):
    """返回 (摆正后的图, side, method, 是否转了 180°, OCR 结果或 None)。OCR 结果与朝向无关，可留给核验复用。"""
    flipped = cv2.rotate(card, cv2.ROTATE_180)
    if face_model and os.path.exists(face_model):
        f0, f1 = _face_score(card, face_model), _face_score(flipped, face_model)
        if max(f0, f1) > 0:
            return (card, "front", "face", False, None) if f0 >= f1 else (flipped, "front", "face", True, None)
    e0, e1 = _emblem_score(card), _emblem_score(flipped)
    if max(e0, e1) > 0:
        return (card, "back", "emblem", False, None) if e0 >= e1 else (flipped, "back", "emblem", True, None)
    lines = _ocr_lines(card) if use_ocr else None
    if lines:
        upright, side = _ocr_orientation(lines, card.shape[0])
        if upright is not None:
            return (card, side, "ocr", False, lines) if upright else (flipped, side, "ocr", True, lines)
    return card, "unknown", "none", False, lines


def orient_card(card: np.ndarray, face_model: Optional[str] = DEFAULT_FACE_MODEL,
                use_ocr: bool = True) -> Tuple[np.ndarray, str, str, bool]:
    """card 为横版，返回 (摆正后的图, side, method, 是否旋转了 180°)。"""
    return _orient(card, face_model, use_ocr)[:4]


# --------------------------------------------------------------------------- 5. 质量检查
def _quality(card: np.ndarray) -> Tuple[float, float]:
    """返回 (清晰度, 最大反光斑占卡面比例)。
    反光斑 = 接近过曝、低饱和的连通块；卡面整体已经很亮（中位亮度 ≥240）时无法区分反光，记为 0。"""
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    inner = gray[int(0.06 * h):int(0.94 * h), int(0.04 * w):int(0.96 * w)]
    sharp = float(cv2.Laplacian(cv2.resize(inner, (856, 460), interpolation=cv2.INTER_AREA), cv2.CV_64F).var())
    hsv = cv2.cvtColor(cv2.resize(card, (428, 270), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    v, s = hsv[:, :, 2], hsv[:, :, 1]
    if np.median(v) >= 240:
        return sharp, 0.0
    glare = cv2.morphologyEx(((v >= 250) & (s <= 40)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(glare)
    biggest = stats[1:, cv2.CC_STAT_AREA].max() if n > 1 else 0
    return sharp, float(biggest) / (428 * 270)


def round_corners(card: np.ndarray, fill=(255, 255, 255)) -> np.ndarray:
    """可选：把四个圆角外的背景涂成纯色。"""
    h, w = card.shape[:2]
    r = int(round(CORNER_R_MM / CARD_H_MM * h))
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (r, 0), (w - 1 - r, h - 1), 255, -1)
    cv2.rectangle(mask, (0, r), (w - 1, h - 1 - r), 255, -1)
    for cx, cy in ((r, r), (w - 1 - r, r), (w - 1 - r, h - 1 - r), (r, h - 1 - r)):
        cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)
    out = card.copy()
    out[mask < 128] = fill
    return out


# --------------------------------------------------------------------------- 总入口
def process_idcard(img: np.ndarray, out_width: int = 1011, expected_side: Optional[str] = None,
                   face_model: Optional[str] = DEFAULT_FACE_MODEL, use_ocr: bool = True,
                   min_detect_score: float = 0.30, confirm_score: float = 0.45,
                   verify: bool = False) -> IdCardResult:
    """
    img              : BGR 图（cv2.imread 读入）
    out_width        : 输出宽度，高度按 85.6:54 计算；1011 ≈ 300dpi 实物尺寸
    expected_side    : 'front' / 'back' / None，前端知道当前在拍哪一面时传入，用于提示拍错面
    min_detect_score : 定位置信度低于它直接判失败（提示重拍）
    confirm_score    : 介于两者之间时照常输出，但 need_confirm=True，建议让用户确认
    verify           : 对每张卡跑一次 OCR，核验确实是身份证的那一面（需安装 rapidocr_onnxruntime，每张多 0.3~0.5 秒）。
                       输入里可能混有行驶证、驾驶证、护照等其他证件时必须打开。结论写在 doc_check / doc_evidence：
                       rejected → ok=False（image 仍保留，便于排查）；uncertain → need_confirm=True，应转人工复核。
    """
    t0 = time.time()
    res = IdCardResult(ok=False)
    out_h = int(round(out_width / CARD_RATIO))

    det = detect_card(img)
    res.detect_score = det.score
    if det.quad is None or det.score < min_detect_score:
        res.warnings.append("未找到完整的身份证，请将卡片完整放入画面、背景尽量与卡片有反差")
        res.elapsed_ms = (time.time() - t0) * 1000
        return res

    q = refine_quad(img, det.quad)
    if _score_quad(q * det.scale, det.feat) < 0.8 * det.score:   # 精修反而变差，回退
        q = det.quad
    card, q_src = warp_card(img, q, out_width, out_h)
    card, side, method, flipped, lines = _orient(card, face_model, use_ocr or verify)
    if flipped:
        q_src = np.roll(q_src, 2, axis=0)

    res.ok, res.image, res.side, res.orient_method, res.quad = True, card, side, method, q_src
    res.sharpness, res.glare_ratio = _quality(card)
    if verify:
        if lines is None:
            lines = _ocr_lines(card)
        if lines is None:
            res.doc_check, res.need_confirm = "unavailable", True
            res.warnings.append("未安装 rapidocr_onnxruntime，无法核验证件类型")
        else:
            res.doc_check, text_side, res.doc_evidence = check_document(_ocr_text(lines), side)
            if res.doc_check == "rejected":   # 不是身份证：后面关于方向、拍错面的提示都没有意义了
                res.ok = False
                res.warnings.append("不是身份证（%s）" % res.doc_evidence)
                res.elapsed_ms = (time.time() - t0) * 1000
                return res
            if res.doc_check == "uncertain":
                res.need_confirm = True
                res.warnings.append("未能确认是身份证：%s，请人工复核" % res.doc_evidence)
            elif side == "unknown":
                res.side = side = text_side
    if det.score < confirm_score:
        res.need_confirm = True
        res.warnings.append("卡片边缘不够清晰，请确认裁剪是否完整（深色纯色背景效果最好）")
    if method == "none":
        res.need_confirm = True
        res.warnings.append("无法判断正反方向，请人工确认")
    if expected_side and side != "unknown" and side != expected_side:
        res.warnings.append("拍到的似乎是%s，请翻面" % ("人像面" if side == "front" else "国徽面"))
    if res.sharpness < 60:
        res.warnings.append("图像可能模糊，建议重拍")
    if res.glare_ratio > 0.02:
        res.warnings.append("卡面有明显反光，建议调整角度重拍")
    res.elapsed_ms = (time.time() - t0) * 1000
    return res


def draw_debug(img: np.ndarray, res: IdCardResult) -> np.ndarray:
    """原图上画出检测框（绿色，红点为摆正后的左上角）+ 右侧拼上结果。"""
    s = 900.0 / max(img.shape[:2])
    vis = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if res.quad is not None:
        q = (res.quad * s).astype(np.int32)
        cv2.polylines(vis, [q.reshape(-1, 1, 2)], True, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(vis, tuple(int(v) for v in q[0]), 7, (0, 0, 255), -1)
    if res.image is not None:
        c = res.image
        cs = min(vis.shape[0] / 2.0 / c.shape[0], 900.0 / c.shape[1])
        c = cv2.resize(c, None, fx=cs, fy=cs, interpolation=cv2.INTER_AREA)
        pad = np.full((vis.shape[0], c.shape[1], 3), 255, np.uint8)
        pad[:c.shape[0]] = c
        vis = np.hstack([vis, pad])
    return vis


def _imread(path: str) -> Optional[np.ndarray]:
    data = np.fromfile(path, dtype=np.uint8)  # 兼容中文路径
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def _imwrite(path: str, img: np.ndarray, quality: int = 95) -> None:
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        buf.tofile(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="身份证照片裁剪 + 摆正（离线 CPU）")
    ap.add_argument("inputs", nargs="+", help="图片文件或目录")
    ap.add_argument("-o", "--out", default="out", help="输出目录（默认 ./out）")
    ap.add_argument("--width", type=int, default=1011, help="输出宽度，默认 1011（300dpi 实物尺寸）")
    ap.add_argument("--side", choices=["front", "back"], help="期望的面（用于拍错面提示）")
    ap.add_argument("--round-corners", action="store_true", help="圆角外背景涂白")
    ap.add_argument("--no-ocr", action="store_true", help="禁用 OCR 兜底")
    ap.add_argument("--verify", action="store_true", help="用 OCR 核验确实是身份证（需 rapidocr_onnxruntime；混有其他证件时必开）")
    ap.add_argument("--debug", action="store_true", help="额外输出带检测框的对照图")
    args = ap.parse_args(argv)

    files = []
    for p in args.inputs:
        if os.path.isdir(p):
            files += sorted(os.path.join(p, f) for f in os.listdir(p)
                            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")))
        else:
            files.append(p)
    os.makedirs(args.out, exist_ok=True)
    n_fail = 0
    for f in files:
        img = _imread(f)
        if img is None:
            print(json.dumps({"file": f, "ok": False, "warnings": ["无法读取图片"]}, ensure_ascii=False))
            n_fail += 1
            continue
        res = process_idcard(img, args.width, args.side, use_ocr=not args.no_ocr, verify=args.verify)
        stem = os.path.splitext(os.path.basename(f))[0]
        if res.ok:
            out = round_corners(res.image) if args.round_corners else res.image
            _imwrite(os.path.join(args.out, stem + "_card.jpg"), out)
        else:
            n_fail += 1
        if args.debug:
            _imwrite(os.path.join(args.out, stem + "_debug.jpg"), draw_debug(img, res), 90)
        print(json.dumps(dict(file=f, **res.summary()), ensure_ascii=False))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
