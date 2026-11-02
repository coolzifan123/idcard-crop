#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合成测试集生成器（仅开发自测用，生产环境不需要）

素材（自行下载，不随代码分发）：
  --template  github.com/airob0t/idcardgenerator 的 idcardgenerator/usedres/empty.png
              （A4 页面：上半张人像面、下半张国徽面，均为空白模板）
  --fontdir   同目录下的 hei.ttf / fzhei.ttf / ocrb10bt.ttf
  --faces     合成人脸目录，例如 HuggingFace bitmind/SyntheticFacesHQ 的 tiny-sample.zip（StyleGAN 生成，非真人）

所有姓名、地址、号码均随机编造，号码以 99 开头（不存在的行政区划），卡面加"合成样本"水印。

输出：
  <out>/scenes/xxxx.jpg  模拟拍摄的照片
  <out>/labels.json      每张图：卡片四角真值（按卡片正向的 左上/右上/右下/左下）、面别、场景参数
"""
import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CARD_W, CARD_H = 1712, 1080          # 20 px/mm
FRONT_BOX = (287, 490, 1886, 1189)   # 模板页面中人像面位置 x,y,w,h
BACK_BOX = (282, 1908, 1887, 1188)   # 国徽面位置

SURNAMES = "王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾肖田董袁潘蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤"
GIVEN = "伟芳娜敏静丽强磊军洋勇艳杰涛明超秀霞平刚桂英华建国志文斌宇浩然欣怡子轩梓涵思远晨阳佳琪雨萱俊杰嘉怡泽宇一诺博文雅婷天佑若曦振华海燕红梅晓东立新春生秋菊"
PROVINCES = [("江西省", ["南昌市", "九江市", "赣州市"], ["东湖区", "西湖区", "青山湖区", "章贡区"]),
             ("浙江省", ["杭州市", "宁波市", "温州市"], ["西湖区", "上城区", "鄞州区", "鹿城区"]),
             ("广东省", ["广州市", "深圳市", "佛山市"], ["天河区", "越秀区", "南山区", "禅城区"]),
             ("四川省", ["成都市", "绵阳市", "宜宾市"], ["武侯区", "锦江区", "涪城区", "翠屏区"]),
             ("湖南省", ["长沙市", "株洲市", "湘潭市"], ["岳麓区", "芙蓉区", "天元区", "雨湖区"]),
             ("山东省", ["济南市", "青岛市", "烟台市"], ["历下区", "市南区", "芝罘区", "槐荫区"]),
             ("河南省", ["郑州市", "洛阳市", "开封市"], ["金水区", "二七区", "涧西区", "龙亭区"])]
ROADS = ["人民路", "解放路", "中山路", "建设路", "文化路", "胜利路", "和平路", "长江路", "黄河路", "新华街", "光明大道"]
NATIONS = ["汉"] * 12 + ["回", "满", "壮", "苗", "土家", "维吾尔", "蒙古", "藏"]


def rand_identity(rng):
    name = rng.choice(list(SURNAMES)) + "".join(rng.choice(list(GIVEN), size=rng.integers(1, 3)))
    prov, cities, dists = PROVINCES[rng.integers(len(PROVINCES))]
    city, dist = rng.choice(cities), rng.choice(dists)
    addr = prov + city + dist + rng.choice(ROADS) + "%d号" % rng.integers(1, 999)
    if rng.random() < 0.6:
        addr += "%d栋%d单元%d室" % (rng.integers(1, 30), rng.integers(1, 6), rng.integers(101, 2899))
    y, m, d = int(rng.integers(1950, 2006)), int(rng.integers(1, 13)), int(rng.integers(1, 29))
    body = "99%04d%04d%02d%02d%03d" % (rng.integers(0, 9999), y, m, d, rng.integers(0, 999))
    wts = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    chk = "10X98765432"[sum(int(c) * w for c, w in zip(body, wts)) % 11]
    start = int(rng.integers(2006, 2024))
    span = int(rng.choice([5, 10, 20, 20, 0]))
    life = ("%d.%02d.%02d-%s" % (start, m, d, "长期" if span == 0 else "%d.%02d.%02d" % (start + span, m, d)))
    return dict(name=name, sex=str(rng.choice(["男", "女"])), nation=str(rng.choice(NATIONS)),
                year=str(y), mon=str(m), day=str(d), addr=addr, idn=body + chk,
                org=city + "公安局" + (dist + "分局" if rng.random() < 0.7 else ""), life=life)


def render_cards(template, fontdir, face_path, rng):
    """在模板页面上填字、贴头像，返回 (人像面, 国徽面) BGR 图，尺寸 CARD_W x CARD_H。"""
    page = template.copy()
    draw = ImageDraw.Draw(page)
    f = lambda name, size: ImageFont.truetype(os.path.join(fontdir, name), size)
    name_font, other_font = f("hei.ttf", 72), f("hei.ttf", 60)
    date_font, id_font = f("fzhei.ttf", 60), f("ocrb10bt.ttf", 72)
    who = rand_identity(rng)
    black = (0, 0, 0, 255)
    draw.text((630, 690), who["name"], fill=black, font=name_font)
    draw.text((630, 840), who["sex"], fill=black, font=other_font)
    draw.text((1030, 840), who["nation"], fill=black, font=other_font)
    draw.text((630, 980), who["year"], fill=black, font=date_font)
    draw.text((950, 980), who["mon"], fill=black, font=date_font)
    draw.text((1150, 980), who["day"], fill=black, font=date_font)
    addr, loc = who["addr"], 1120
    for s in range(0, len(addr), 11):
        draw.text((630, loc), addr[s:s + 11], fill=black, font=other_font)
        loc += 100
    draw.text((950, 1475), who["idn"], fill=black, font=id_font)
    draw.text((1050, 2750), who["org"], fill=black, font=other_font)
    draw.text((1050, 2895), who["life"], fill=black, font=other_font)

    # 头像：合成人脸裁成 500x670，边缘羽化后贴到 (1500, 690)
    face = Image.open(face_path).convert("RGB")
    fw, fh = face.size
    cw = int(fh * 500 / 670)
    face = face.crop(((fw - cw) // 2, 0, (fw - cw) // 2 + cw, fh)).resize((500, 670), Image.LANCZOS)
    alpha = np.ones((670, 500), np.float32)
    ramp = np.linspace(0, 1, 25, dtype=np.float32)
    alpha[:25, :] *= ramp[:, None]
    alpha[-25:, :] *= ramp[::-1, None]
    alpha[:, :25] *= ramp[None, :]
    alpha[:, -25:] *= ramp[None, ::-1]
    page.paste(face, (1500, 690), Image.fromarray((alpha * 255).astype(np.uint8)))

    # 水印
    wm = Image.new("RGBA", page.size, (0, 0, 0, 0))
    ImageDraw.Draw(wm).text((1180, 1290), "合成样本 SYNTHETIC", fill=(120, 120, 120, 110), font=f("hei.ttf", 44))
    ImageDraw.Draw(wm).text((1480, 3010), "合成样本 SYNTHETIC", fill=(120, 120, 120, 110), font=f("hei.ttf", 44))
    page = Image.alpha_composite(page, wm)

    arr = cv2.cvtColor(np.asarray(page.convert("RGB")), cv2.COLOR_RGB2BGR)
    out = []
    for x, y, w, h in (FRONT_BOX, BACK_BOX):
        card = arr[y:y + h, x:x + w]
        out.append(cv2.resize(card, (CARD_W, CARD_H), interpolation=cv2.INTER_AREA))
    return out[0], out[1], who


def rounded_mask(w, h, r):
    m = np.zeros((h, w), np.uint8)
    cv2.rectangle(m, (r, 0), (w - 1 - r, h - 1), 255, -1)
    cv2.rectangle(m, (0, r), (w - 1, h - 1 - r), 255, -1)
    for cx, cy in ((r, r), (w - 1 - r, r), (w - 1 - r, h - 1 - r), (r, h - 1 - r)):
        cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return m


# ------------------------------------------------------------------ 背景
def smooth_noise(rng, w, h, scale, amp):
    small = rng.normal(0, 1, (max(2, h // scale), max(2, w // scale))).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) * amp


def make_background(rng, w, h):
    kind = str(rng.choice(["plain_light", "plain_dark", "wood", "fabric", "paper", "similar", "clutter", "gradient"]))
    if kind == "plain_light":
        base = rng.uniform(170, 245, 3)
        bg = np.ones((h, w, 3), np.float32) * base
    elif kind == "plain_dark":
        base = rng.uniform(15, 90, 3)
        bg = np.ones((h, w, 3), np.float32) * base
    elif kind == "wood":
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ang = rng.uniform(0, np.pi)
        t = xx * np.cos(ang) + yy * np.sin(ang)
        warp = smooth_noise(rng, w, h, 60, 25)
        grain = 0.5 + 0.5 * np.sin((t + warp) / rng.uniform(6, 18))
        c1, c2 = np.array([40, 75, 120]) * rng.uniform(0.7, 1.4), np.array([70, 120, 175]) * rng.uniform(0.7, 1.3)
        bg = c1[None, None, :] * grain[..., None] + c2[None, None, :] * (1 - grain[..., None])
    elif kind == "fabric":
        yy, xx = np.mgrid[0:h, 0:w]
        p = int(rng.integers(4, 14))
        chk = (((xx // p) + (yy // p)) % 2).astype(np.float32)
        c1, c2 = rng.uniform(30, 230, 3), rng.uniform(30, 230, 3)
        bg = c1 * chk[..., None] + c2 * (1 - chk[..., None])
        bg = cv2.GaussianBlur(bg, (0, 0), 0.8)
    elif kind == "paper":  # 卡片压在一张印着字的白纸上
        bg = np.ones((h, w, 3), np.float32) * rng.uniform(215, 250)
        for y in range(int(rng.integers(20, 60)), h, int(rng.integers(28, 48))):
            x = int(rng.integers(30, 120))
            while x < w - 60:
                L = int(rng.integers(15, 160))
                cv2.rectangle(bg, (x, y), (min(w - 40, x + L), y + int(rng.integers(8, 16))),
                              tuple(float(v) for v in rng.uniform(40, 120, 3)), -1)
                x += L + int(rng.integers(10, 30))
    elif kind == "similar":  # 和卡面颜色相近的浅蓝/浅灰（最难）
        bg = np.ones((h, w, 3), np.float32) * np.array([rng.uniform(205, 240), rng.uniform(200, 230), rng.uniform(185, 220)])
    elif kind == "clutter":
        bg = np.ones((h, w, 3), np.float32) * rng.uniform(80, 220, 3)
        for _ in range(int(rng.integers(5, 15))):
            col = tuple(float(v) for v in rng.uniform(0, 255, 3))
            if rng.random() < 0.5:
                p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
                p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
                cv2.line(bg, p1, p2, col, int(rng.integers(3, 25)))
            else:
                c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
                cv2.ellipse(bg, c, (int(rng.integers(20, 200)), int(rng.integers(20, 200))),
                            float(rng.uniform(0, 180)), 0, 360, col, -1)
    else:  # gradient
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ang = rng.uniform(0, 2 * np.pi)
        t = (xx * np.cos(ang) + yy * np.sin(ang)) / max(w, h)
        a, b = rng.uniform(30, 250, 3), rng.uniform(30, 250, 3)
        bg = a[None, None, :] * (1 - t[..., None]) + b[None, None, :] * t[..., None]
    bg = bg + smooth_noise(rng, w, h, 80, rng.uniform(2, 10))[..., None]
    return np.clip(bg, 0, 255), kind


# ------------------------------------------------------------------ 场景合成
def make_scene(card, rng):
    sizes = [(1600, 1200), (1200, 1600), (2000, 1500), (1500, 2000), (1308, 2000), (1920, 1080), (1080, 1920), (3024, 4032)]
    W, H = sizes[rng.integers(len(sizes))]
    bg, bg_kind = make_background(rng, W, H)
    rot90 = int(rng.integers(0, 4))
    tilt = float(rng.uniform(-12, 12))
    frac = float(rng.uniform(0.12, 0.85))          # 卡片占画面面积比例
    persp = float(rng.uniform(0.0, 0.06))          # 透视强度
    for _ in range(50):
        cw = np.sqrt(frac * W * H * CARD_W / CARD_H)
        ch = cw * CARD_H / CARD_W
        pts = np.float32([[-cw / 2, -ch / 2], [cw / 2, -ch / 2], [cw / 2, ch / 2], [-cw / 2, ch / 2]])
        pts += rng.uniform(-persp, persp, (4, 2)).astype(np.float32) * ch
        th = np.radians(rot90 * 90 + tilt)
        R = np.float32([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        pts = pts @ R.T
        margin = 0.015 * min(W, H)
        span = pts.max(0) - pts.min(0)
        if span[0] < W - 2 * margin and span[1] < H - 2 * margin:
            break
        frac *= 0.9
    lo = -pts.min(0) + margin
    hi = np.float32([W, H]) - pts.max(0) - margin
    pts += np.float32([rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1])])

    src = np.float32([[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]])
    M = cv2.getPerspectiveTransform(src, pts.astype(np.float32))
    r = int(round(3.18 / 54.0 * CARD_H))
    mask = rounded_mask(CARD_W, CARD_H, r)
    warped = cv2.warpPerspective(card, M, (W, H), flags=cv2.INTER_LINEAR).astype(np.float32)
    m = cv2.warpPerspective(mask, M, (W, H), flags=cv2.INTER_LINEAR).astype(np.float32) / 255.0

    # 阴影
    sh = cv2.GaussianBlur(m, (0, 0), rng.uniform(3, 15) * W / 1600)
    sh = np.roll(sh, (int(rng.integers(-8, 9)), int(rng.integers(-8, 9))), axis=(0, 1))
    img = bg * (1 - rng.uniform(0.15, 0.5) * sh[..., None])
    # 卡片本身亮度/色温略有变化（不同相机、不同光源）
    card_gain = rng.uniform(0.85, 1.08) * rng.uniform(0.94, 1.06, 3)
    img = img * (1 - m[..., None]) + warped * card_gain * m[..., None]

    extras = []
    if rng.random() < 0.15:  # 反光
        c = pts.mean(0) + rng.uniform(-0.3, 0.3, 2) * (pts.max(0) - pts.min(0))
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        s = rng.uniform(0.03, 0.08) * min(W, H)
        g = np.exp(-(((xx - c[0]) / (s * rng.uniform(1, 2.5))) ** 2 + ((yy - c[1]) / s) ** 2))
        img = img + g[..., None] * m[..., None] * rng.uniform(80, 200)
        extras.append("glare")
    if rng.random() < 0.15:  # 手指压住一条边
        i = int(rng.integers(0, 4))
        a, b = pts[i], pts[(i + 1) % 4]
        c = a + rng.uniform(0.25, 0.75) * (b - a)
        fl = 0.09 * min(W, H)
        ang = np.degrees(np.arctan2(*(b - a)[::-1])) + 90 + rng.uniform(-20, 20)
        skin = (float(rng.uniform(120, 170)), float(rng.uniform(150, 190)), float(rng.uniform(200, 235)))
        cv2.ellipse(img, (int(c[0]), int(c[1])), (int(fl), int(fl * 0.42)), float(ang), 0, 360, skin, -1, cv2.LINE_AA)
        extras.append("finger")

    # 光照不均
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    t = (xx * np.cos(ang) + yy * np.sin(ang)) / max(W, H)
    light = 1 + rng.uniform(-0.25, 0.25) * (t - t.mean())
    img = img * light[..., None]
    # 相机：模糊、噪声、JPEG
    blur = float(rng.choice([0, 0, 0.6, 1.0, 1.6, 2.4])) * W / 1600
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    img = img + rng.normal(0, rng.uniform(0, 6), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    q = int(rng.integers(55, 96))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    meta = dict(size=[W, H], bg=bg_kind, rot90=rot90 * 90, tilt=round(tilt, 1), area_frac=round(frac, 3),
                persp=round(persp, 3), blur=round(blur, 2), jpeg=q, extras=extras)
    return buf, pts, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--fontdir", required=True)
    ap.add_argument("--faces", required=True, help="人脸目录")
    ap.add_argument("--face-list", help="可选：json 文件，只用其中列出的人脸文件名")
    ap.add_argument("--out", required=True)
    ap.add_argument("-n", type=int, default=200, help="场景图数量（正反面各半）")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--save-cards", action="store_true", help="同时保存平面卡片图")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    template = Image.open(args.template).convert("RGBA")
    faces = json.load(open(args.face_list)) if args.face_list else sorted(os.listdir(args.faces))
    os.makedirs(os.path.join(args.out, "scenes"), exist_ok=True)
    if args.save_cards:
        os.makedirs(os.path.join(args.out, "cards"), exist_ok=True)
    labels = []
    for i in range(0, args.n, 2):
        face = os.path.join(args.faces, faces[(i // 2) % len(faces)])
        front, back, _ = render_cards(template, args.fontdir, face, rng)
        if args.save_cards:
            cv2.imwrite(os.path.join(args.out, "cards", "%04d_front.png" % i), front)
            cv2.imwrite(os.path.join(args.out, "cards", "%04d_back.png" % i), back)
        for side, card, idx in (("front", front, i), ("back", back, i + 1)):
            buf, pts, meta = make_scene(card, rng)
            name = "%04d_%s.jpg" % (idx, side)
            buf.tofile(os.path.join(args.out, "scenes", name))
            labels.append(dict(file=name, side=side, corners=np.round(pts, 2).tolist(), **meta))
        print("\r%d/%d" % (min(i + 2, args.n), args.n), end="", flush=True)
    json.dump(labels, open(os.path.join(args.out, "labels.json"), "w"), ensure_ascii=False, indent=1)
    print("\nsaved", len(labels), "scenes to", args.out)


if __name__ == "__main__":
    main()
