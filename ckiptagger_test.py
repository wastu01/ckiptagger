# -*- coding: utf-8 -*-

# 移除警告提示訊息
import os
from tensorflow.python.util import deprecation
import tensorflow as tf
os.environ["TF_USE_LEGACY_KERAS"]='1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# 導入模組
from ckiptagger import data_utils
from ckiptagger import WS, POS, NER

# 下載模型檔案
# data_utils.download_data_gdown("./")

text = '傅達仁今將執行安樂死，卻突然爆出自己20年前遭緯來體育台封殺，他不懂自己哪裡得罪到電視台。'

# 初始化
ws = WS("./data")
pos = POS("./data")
ner = NER("./data")

# 斷詞
ws_results = ws([text])
# 詞性標註
pos_results = pos(ws_results)
# 命名實體辨識
ner_results = ner(ws_results, pos_results)


# 注意資料型態是二維陣列
# print(ws_results)
# print(ws_results[0])
print(ws_results[0][0])

# print(pos_results)
# print(pos_results[0])
print(pos_results[0][0])

print(ner_results)
print(type(ner_results[0]))



# for name in ner_results[0]:
#     print(name)