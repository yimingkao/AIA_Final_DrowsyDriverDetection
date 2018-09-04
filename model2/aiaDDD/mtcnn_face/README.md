# YawDD - MTCNN特徵點框嘴巴區域

先前已經在 SSD 的實驗決定要抽的 feature 個數為 512. 這邊只是使用 MTCNN 來把 feature 點找出來框出嘴巴, 然後抽 YawDD 的 features.
這邊主要使用原先 training 過的 mobilenet 跟自己做的 CNN 來抽 feature.

## 目錄結構

| Name | Description |
| ---- | -------- |
| bbox | YawDD 使用 MTCNN 找到的人臉及特徵位置 |
| config.py | 設定要選擇的抽 feature algorithm |
| MTCNN | MTCNN 的實作 folder |
| mtcnn\_face\_det.py | MTCNN 的人臉偵測 class |
| mtcnn2bbox.py | 將每個 video 由 MTCNN 的人臉偵測結果寫到 bbox 裡的一個個 csv 檔 |
| bbox2fea.py | 每個 video 每張 frame 抽取預定義位置的嘴巴區域的 feature 數並存成 .npy 檔 |
| verify.py | 將MTCNN找到的特徵點框嘴巴區域畫在 video 上來驗證位置是否正確 |
| mobilenet\_custom\_fea\_extract.py | Transfer learning 過的 mobilenet 抽 feature 的 class code |
| mobilecus\_fea\_512.h5 | mobilenet\_custom\_fea\_extract.py 的原始模型檔 |
| mouthnnym\_fea\_extract.py | 縮減 mobilenet 而自定的 CNN feature extractor |
| mouthnnym\_saved | mouthnnym\_fea\_extract.py 的模型檔 |

## 操作順序

```
$ python mtcnn2bbox.py
$ python verify.py
$ python bbox2fea.py (會使用到 mobilenet_custom_feature_extract.py 或是 mouthnnym_fea_extract.py)
```
