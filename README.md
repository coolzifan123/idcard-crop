# 身份证照片自动裁剪 + 摆正（离线 / 纯 CPU）

拍摄的二代身份证照片（人像面或国徽面）→ 自动找到卡片、裁掉背景、透视拉正、文字转正，输出标准尺寸的卡面图。

- **不需要 GPU，不联网**：只依赖 OpenCV + numpy，外加一个 227KB 的人脸检测模型（YuNet，MIT 协议）
- **速度**：4 核约 150ms/张，单线程约 200ms/张（1200 万像素照片单线程约 310ms）
- **输出**：默认 1011×638，即 300dpi 下的实物尺寸（85.6×54mm），打印出来就是 1:1 大小

![合成数据效果示例](docs/demo_synthetic.jpg)

## 目录

```
idcard_crop/
├── idcard_crop.py                          核心代码，可 import，也可命令行调用
├── models/face_detection_yunet_2023mar.onnx 人脸检测模型（OpenCV Zoo，MIT，许可证见 models/LICENSE）
├── requirements.txt
├── tools/synth_idcards.py                  合成测试数据生成器（只在开发自测时用）
├── tools/evaluate.py                       评测脚本
└── docs/demo_synthetic.jpg                 效果示例（全部为合成数据）
```

## 处理流程

```
照片 ──► ① 粗定位（缩到 480px）──► ② 原图精修 ──► ③ 透视矫正 ──► ④ 方向判断 ──► ⑤ 质量检查 ──► 卡面图 + JSON
         彩色边缘 + 霍夫直线          逐点找边 +       长边→宽        只剩 0°/180°     模糊 / 反光 /
         四边形候选打分               RANSAC 拟合       1011×638       人脸 / 国徽 / OCR   置信度
```

1. **粗定位**：把彩色边缘按梯度方向分成"近水平 / 近竖直"两组分别做霍夫直线，两两组合成四边形（上万种组合向量化一次评估），再按下面几项打分取最优：
   - 四条边的**边缘支撑度**：卡片内外两侧色差方向沿整条边一致（文字、纹理处方向随机，拿不到高分），四条边取几何平均，任何一条边不像卡边都会重罚；
   - **长宽比**接近 85.6:54（±9% 内不扣分，以容忍透视）；
   - **内侧边距均匀**：身份证四周有一圈空白边距，卡边往里一点颜色很均匀，用来区分卡片和文字块、头像框之类；
   - 最弱一条边的对比强度（压制阴影外缘）、面积（同等证据下取最外层的矩形）。
2. **精修**：回到原图分辨率，沿每条边的法线方向找边缘点，先宽后窄两轮，RANSAC 拟合直线后求交点。身份证是圆角，用直线交点得到"虚拟直角"，拉正后四边严格对齐。选边缘点时优先选"内侧颜色像卡片边距"的那条，避免落到阴影外缘上。
3. **透视矫正**：长边映射为宽。竖着拍的卡片在这一步自动转成横版，之后只剩 0° 和 180° 两种可能。
4. **方向判断**：
   - 人像面：YuNet 人脸检测，人脸在右侧、眼睛在嘴上方即为正向；
   - 国徽面：国徽（大小、形状都对得上的红色色块）应在左上角；
   - 兜底（可选）：装了 RapidOCR 时，找"姓名 / 公民身份号码 / 签发机关 / 有效期限"等固定字样，看它们在上还是在下（头像被反光盖住时有用）。
5. **质量检查**：清晰度（拉普拉斯方差）、反光斑面积、定位置信度，供前端决定"直接用 / 请用户确认 / 请重拍"。

## 测试结果

手头没有真实证件样本，所以用开源模板做了一套**合成测试集**：卡面用 [airob0t/idcardgenerator](https://github.com/airob0t/idcardgenerator) 的空白模板，头像用 [SyntheticFacesHQ](https://huggingface.co/datasets/bitmind/SyntheticFacesHQ)（StyleGAN 生成，都不是真人），姓名、地址、号码随机编造（号码以 99 开头，不对应任何真实行政区划），卡面加了"合成样本"水印。然后贴到 8 类背景上，随机加 0/90/180/270° 旋转 ±12° 倾斜、透视、阴影、反光（15%）、手指遮挡（15%）、模糊、噪声、JPEG 压缩，分辨率从 1080p 到 1200 万像素。

| 指标 | 调参集 300 张 | 留出集 200 张（未参与调参） |
|---|---|---|
| 裁剪正确（角点误差 < 卡长 2%） | 92.3% | 92.0% |
| └ 纯色 / 木纹 / 格子布 / 杂物 / 渐变 / 近似色背景 | 267/267 | 179/180 |
| └ 印满字的白纸背景 | 30% | 25% |
| 方向正确（在裁剪正确的样本里） | 100% | 100% |
| 正反面识别正确 | 100% | 100% |
| 平均角点误差 | 0.41% 卡长 | 0.39% 卡长 |

置信度分档（留出集）：`detect_score ≥ 0.45` 时输出的图 96.8% 正确，错误大多落在 0.30~0.45 的"请确认"档或 < 0.30 的"请重拍"档。纸张背景下卡边紧贴文字，是目前的主要短板，所以前端要提示用户放在纯色桌面上拍。

## 离线部署

1. 在一台**能上网、操作系统和 Python 版本都与目标机相同**的电脑上下载依赖：

```bash
pip download -d wheels numpy opencv-python-headless
```

   需要 OCR 兜底的话再加上 `rapidocr_onnxruntime`（会连带下载 onnxruntime 等，模型已打包在 pip 包里）。

2. 把 `wheels/` 和整个 `idcard_crop/` 目录拷到目标机，然后安装：

```bash
pip install --no-index --find-links wheels numpy opencv-python-headless
```

要求 Python ≥ 3.8、OpenCV ≥ 4.8（YuNet 2023mar 模型需要 4.8+），已在 OpenCV 4.10 和 5.0 上验证结果一致。

## 使用

命令行（可传多个文件或目录，每张图输出一行 JSON）：

```bash
python idcard_crop.py 照片1.jpg 照片2.jpg -o out --debug
```

```bash
python idcard_crop.py 照片目录/ -o out --side front --round-corners
```

- `--side front|back`：告诉程序当前在拍哪一面，拍反了会提示
- `--round-corners`：把四个圆角外的背景涂白
- `--width 1011`：输出宽度（高度按比例算）
- `--debug`：额外输出带检测框的对照图

Python 调用：

```python
import cv2
from idcard_crop import process_idcard

res = process_idcard(cv2.imread("photo.jpg"), expected_side="front")
if not res.ok:
    print("请重拍：", res.warnings)
elif res.need_confirm:
    show_to_user(res.image, res.warnings)   # 展示结果让用户确认
else:
    cv2.imwrite("card.jpg", res.image)
```

返回字段：

| 字段 | 说明 |
|---|---|
| `ok` | 是否得到卡面图；False 时应提示重拍 |
| `need_confirm` | 定位置信度偏低或方向没把握，建议展示给用户确认 |
| `image` | 裁剪摆正后的 BGR 图 |
| `side` | `front` 人像面 / `back` 国徽面 / `unknown` |
| `orient_method` | 方向依据：`face` / `emblem` / `ocr` / `none` |
| `quad` | 卡片在原图中的四角（按摆正后的左上、右上、右下、左下） |
| `detect_score` | 定位置信度 0~1（< 0.30 失败，0.30~0.45 需确认） |
| `sharpness` / `glare_ratio` | 清晰度 / 最大反光斑占卡面比例 |
| `warnings` | 给用户看的中文提示 |

检测器按线程缓存，可以直接在多线程服务里调用。

## 拍摄端建议（对成功率影响最大）

- 卡片放在**深色、纯色**的桌面或垫子上，不要放在印满字的纸上
- 卡片完整入画，最好占画面 1/4 以上；横拍竖拍都可以，稍微歪一点没关系
- 避开顶灯直射造成的反光，等对焦清楚再拍
- 用高拍仪的话，底板选深色效果最好；如果有取景框，引导用户把卡放进框里

## 已知局限与后续

- **印满字的纸张背景**准确率低（见上表），大多会落到"请确认 / 请重拍"。
- **卡边极低对比**时可能残留一条细背景边。你给的样图就是这种情况：卡片左边缘外侧紧挨着一条和卡片同样亮的阴影带。目前的结果是方向正确、背景基本裁掉，底边残留一条细边，并标记了 `need_confirm`。
- 倾斜超过约 25° 或透视很强的照片不在设计范围内（需求说明里不会太斜）。
- 合成数据不等于真实数据。上线前建议用实际设备拍 50~100 张真实照片，校准 `confirm_score`（0.45）、清晰度阈值（60）、反光阈值（0.02）。
- **升级路线**：如果真实场景里还是不够准，可以用 `tools/synth_idcards.py` 在有 GPU 的机器上批量生成数据，训练一个小型分割或角点模型（如 MobileNetV3 + U-Net，256×256 输入），导出 ONNX 后用 `cv2.dnn` 在 CPU 上推理（约 20~40ms），替换第 ① 步；精修和方向判断不用改。
- **其他语言**：OpenCV 在 C++ / Java / C#（OpenCvSharp）里的 API 基本一致，核心代码约 800 行（含注释和命令行），可以直接移植；也可以把这个脚本包成本机 HTTP 服务来调用。

## 重新生成合成测试集

素材本身不随代码分发，需要自己下载（`docs/demo_synthetic.jpg` 是用这些素材合成的示例图）：

- 卡面模板和字体：[airob0t/idcardgenerator](https://github.com/airob0t/idcardgenerator)（GPL-3.0）里的 `idcardgenerator/usedres/`（`empty.png`、`hei.ttf`、`fzhei.ttf`、`ocrb10bt.ttf`）
- 合成人脸：[bitmind/SyntheticFacesHQ](https://huggingface.co/datasets/bitmind/SyntheticFacesHQ) 的 `tiny-sample.zip`，解压到一个目录（商用前请自行确认许可）

```bash
pip install pillow
```

```bash
python tools/synth_idcards.py --template usedres/empty.png --fontdir usedres --faces faces/ --out synth -n 300 --seed 2026
```

```bash
python tools/evaluate.py synth --fail-dir synth_fail
```
