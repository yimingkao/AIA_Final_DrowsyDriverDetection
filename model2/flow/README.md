# 呵欠程度偵測 Demo 流程 - SSD Face Detect


## 目錄結構
| Name | Description |
| ---- | -------- |
| dense121_fea_extract.py | Dense121 feature extractor |
| mobilenet_fea_extract.py | Mobilenet (with imagenet weighting) feature extractor |
| mobilenet_custom_fea_extract.py | Mobilenet (trained) feature extractor |
| end2end.py | 從 sensor 進 frame 到 SSD 找臉, 框出嘴巴區域後抽 feature, 再丟給 LSTM 判斷 |
| end2end_mp.py | 同上, 找臉跟抽 feature 改為兩個 task 在跑, 速度快一些. |
| end2end_aiaddd.py | 對於 aiaDDD 做整個流程的驗證 |
| end2end_yawdd.py | 對於 YawDD 做整個流程的驗證 |

