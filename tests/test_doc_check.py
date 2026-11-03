# -*- coding: utf-8 -*-
"""证件核验规则的单元测试：只测文字规则，不需要 OCR 和图片。
运行：python tests/test_doc_check.py（也兼容 pytest）"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from idcard_crop import _ocr_orientation, check_document, find_id_numbers  # noqa: E402

W = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)


def make_id(body17):
    return body17 + "10X98765432"[sum(int(c) * w for c, w in zip(body17, W)) % 11]


ID = make_id("11010519491231002")   # GB 11643 里的示例号码，校验位为 X
HK = make_id("81000019900101123")   # 港澳台居民居住证号码（81 开头）

FRONT = "姓名|王小明|性别男|民族汉|出生1990年1月1日|住址|某省某市某区某路1号|公民身份号码|" + ID
BACK = "中华人民共和国|居民身份证|签发机关|某市公安局|有效期限|2016.05.01-2026.05.01"
XINGSHI = ("中华人民共和国机动车行驶证|号牌号码|测A00000|车辆类型|小型普通客车|所有人|测试|"
           "使用性质|非营运|注册日期|2015-06-01")
JIASHI = ("中华人民共和国机动车驾驶证|证号|" + ID + "|姓名|测试|性别|男|国籍|中国|住址|某地|出生日期|1990-01-01|"
          "初次领证日期|2010-01-01|准驾车型|C1|有效期限|2016-01-01至2022-01-01")
PASSPORT = "中华人民共和国|护照|PASSPORT|姓名|测试|签发机关|公安部出入境管理局|有效期至|2030.01.01"
FOREIGNER = "中华人民共和国外国人永久居留身份证|签发机关|国家移民管理局|有效期限|2020.01.01-2030.01.01"
HK_FRONT = "姓名|测试|性别男|出生1990年1月1日|住址|香港|公民身份号码|" + HK


def check(text, side, verdict, out_side=None):
    v, s, ev = check_document(text, side)
    assert v == verdict, (text[:30], side, v, ev)
    if out_side:
        assert s == out_side, (text[:30], s, out_side)
    return ev


def test_id_numbers():
    assert ID.endswith("X") and find_id_numbers("号码" + ID) == [ID]
    assert find_id_numbers(ID.lower()) == [ID]                     # 小写 x
    assert find_id_numbers(ID[:-1] + "1") == []                    # 校验位错
    assert find_id_numbers(make_id("11010519491331002")) == []     # 13 月
    assert find_id_numbers("1" + ID) == []                         # 19 位数字串不算


def test_real_id_card():
    check(FRONT, "front", "verified", "front")
    check(BACK, "back", "verified", "back")
    check(FRONT, "unknown", "verified", "front")                   # 面未知时由文字决定
    check(BACK, "unknown", "verified", "back")


def test_ocr_errors_and_glare():
    check(BACK.replace("居民身份证", "居民身分证"), "back", "verified")                      # OCR 错 1 个字
    check(FRONT.replace("民族汉|", "").replace("公民身份号码", "公民身经号码"), "front", "verified")
    check("签发机关|某市公安局|有效期限|2016.05.01-长期", "back", "verified")               # 标题被反光盖住
    check("中华人民共和国", "back", "uncertain")                                            # 只剩通用字样
    check("姓名王小明|性别男|民族汉", "front", "uncertain")                                  # 有"民族"但号码没认出
    ev = check("王小明|某省某市某区某路1号|" + ID, "front", "uncertain")                     # 重度模糊：只认出号码
    assert "号码" in ev


def test_other_documents_rejected():
    for text, side in ((XINGSHI, "back"), (XINGSHI, "unknown"), (JIASHI, "front"), (JIASHI, "back"),
                       (PASSPORT, "back"), (FOREIGNER, "back"), (HK_FRONT, "front")):
        check(text, side, "rejected")


def test_side_mismatch():
    check(BACK, "front", "uncertain")
    check(FRONT, "back", "uncertain")


def test_ocr_fallback_no_longer_calls_xingshi_back():
    # 旧版 OCR 兜底把"中华人民共和国"当成国徽面字样，行驶证会被判成 back
    lines = [(40.0, "中华人民共和国机动车行驶证"), (160.0, "号牌号码测A00000"), (220.0, "车辆类型小型普通客车")]
    assert _ocr_orientation(lines, 638) == (None, "unknown")
    lines = [(150.0, "中华人民共和国"), (260.0, "居民身份证"), (450.0, "签发机关某市公安局"),
             (540.0, "有效期限2016.05.01-2026.05.01")]
    assert _ocr_orientation(lines, 638) == (True, "back")
    assert _ocr_orientation([(638 - y, t) for y, t in lines], 638) == (False, "back")   # 倒置


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print("ok ", name)
    print("全部通过（%d 组）" % n)
