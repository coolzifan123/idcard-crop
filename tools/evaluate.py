#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在合成测试集上评测：定位精度、方向正确率、正反面识别、耗时。
用法: python tools/evaluate.py <数据目录(含 scenes/ 和 labels.json)> [--fail-dir 失败样本输出目录] [--verify]
--verify 时额外统计证件核验结果（测试集全是身份证：rejected 即误剔除，uncertain 即需人工复核）。
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import idcard_crop as ic  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--tol", type=float, default=0.02, help="角点误差容限（相对卡片长边），默认 2%%")
    ap.add_argument("--fail-dir")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--verify", action="store_true", help="同时评测证件核验（需 rapidocr_onnxruntime）")
    args = ap.parse_args()

    labels = json.load(open(os.path.join(args.data, "labels.json"), encoding="utf-8"))
    if args.limit:
        labels = labels[:args.limit]
    if args.fail_dir:
        os.makedirs(args.fail_dir, exist_ok=True)

    rows = []
    for L in labels:
        img = cv2.imread(os.path.join(args.data, "scenes", L["file"]))
        gt = np.array(L["corners"], np.float32)
        long_side = max(np.linalg.norm(gt[1] - gt[0]), np.linalg.norm(gt[2] - gt[1]))
        t = time.perf_counter()
        res = ic.process_idcard(img, use_ocr=not args.no_ocr, verify=args.verify)
        dt = (time.perf_counter() - t) * 1000
        if res.quad is None:
            err = err_any = float("inf")
        else:
            err = float(np.linalg.norm(res.quad - gt, axis=1).max() / long_side)
            err_any = min(float(np.linalg.norm(np.roll(res.quad, k, 0) - gt, axis=1).max() / long_side)
                          for k in range(4))
        r = dict(file=L["file"], side=L["side"], bg=L["bg"], extras=L["extras"], blur=L["blur"],
                 crop_ok=err_any < args.tol, full_ok=err < args.tol, err=err_any,
                 side_ok=res.side == L["side"], method=res.orient_method, ms=dt, warnings=res.warnings,
                 score=res.detect_score, ok=res.ok, doc=res.doc_check, evidence=res.doc_evidence)
        rows.append(r)
        if args.fail_dir and not r["full_ok"]:
            vis = ic.draw_debug(img, res)
            s = 900.0 / max(img.shape[:2])
            cv2.polylines(vis, [(gt * s).astype(np.int32).reshape(-1, 1, 2)], True, (255, 0, 255), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(args.fail_dir, L["file"]), vis)
        print("\r%d/%d" % (len(rows), len(labels)), end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    def pct(xs):
        return "%5.1f%%" % (100.0 * np.mean(xs)) if len(xs) else "  -  "

    n = len(rows)
    crop = [r["crop_ok"] for r in rows]
    full = [r["full_ok"] for r in rows]
    orient_given_crop = [r["full_ok"] for r in rows if r["crop_ok"]]
    side = [r["side_ok"] for r in rows if r["full_ok"]]
    errs = np.array([r["err"] for r in rows if r["crop_ok"]])
    ms = np.array([r["ms"] for r in rows])
    print("样本数                 %d" % n)
    print("裁剪正确(角点误差<%.0f%%)  %s" % (args.tol * 100, pct(crop)))
    print("裁剪+方向都正确         %s" % pct(full))
    print("方向正确率(裁剪正确时)  %s" % pct(orient_given_crop))
    print("正反面识别正确          %s" % pct(side))
    if len(errs):
        print("角点误差 均值/中位/最大  %.2f%% / %.2f%% / %.2f%%（相对卡片长边）"
              % (errs.mean() * 100, np.median(errs) * 100, errs.max() * 100))
    print("耗时 均值/P95           %.0f ms / %.0f ms" % (ms.mean(), np.percentile(ms, 95)))

    groups = defaultdict(list)
    for r in rows:
        groups["side=" + r["side"]].append(r)
        groups["bg=" + r["bg"]].append(r)
        for e in r["extras"]:
            groups["extra=" + e].append(r)
        groups["blur=%s" % ("strong" if r["blur"] >= 2.0 else "weak/none")].append(r)
        groups["method=" + r["method"]].append(r)
    print("\n%-22s %5s %8s %8s" % ("分组", "数量", "裁剪", "裁剪+方向"))
    for k in sorted(groups):
        g = groups[k]
        print("%-22s %5d %8s %8s" % (k, len(g), pct([r["crop_ok"] for r in g]), pct([r["full_ok"] for r in g])))
    if args.verify:
        good = [r for r in rows if r["full_ok"]]
        print("\n证件核验（只统计裁剪+方向都正确的 %d 张；测试集全是身份证）：" % len(good))
        for k, desc in (("verified", "通过"), ("uncertain", "转人工复核"), ("rejected", "误剔除")):
            sub = [r for r in good if r["doc"] == k]
            print("  %-9s %-10s %4d  %s" % (k, desc, len(sub), pct([1] * len(sub) + [0] * (len(good) - len(sub)))))
        for side in ("front", "back"):
            sub = [r for r in good if r["side"] == side]
            print("  %s 通过率 %s" % (side, pct([r["doc"] == "verified" for r in sub])))
        for r in good:
            if r["doc"] != "verified":
                print("    %s %s blur=%.1f extras=%s  %s" % (r["file"], r["doc"], r["blur"], r["extras"], r["evidence"]))
    print("\n置信度阈值分析（低于阈值 → 提示重拍）：")
    print("%8s %10s %12s %12s" % ("阈值", "输出比例", "输出中正确率", "错误被拦截"))
    for thr in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        out = [r for r in rows if r["ok"] and r["score"] >= thr]
        bad = [r for r in rows if not r["full_ok"]]
        caught = [r for r in bad if not (r["ok"] and r["score"] >= thr)]
        print("%8.2f %10s %12s %12s" % (thr, pct([1] * len(out) + [0] * (n - len(out))),
                                         pct([r["full_ok"] for r in out]),
                                         "%d/%d" % (len(caught), len(bad))))
    fails = [r for r in rows if not r["full_ok"]]
    if fails:
        print("\n失败样本:")
        for r in fails:
            print("  %s bg=%s extras=%s err=%.3f score=%.2f method=%s %s" % (
                r["file"], r["bg"], r["extras"], r["err"], r["score"], r["method"], r["warnings"]))


if __name__ == "__main__":
    main()
