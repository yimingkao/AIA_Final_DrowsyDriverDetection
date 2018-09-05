# LSTM 模型訓練 - MTCNN Face Detect + Trained Mobilenet

這邊就是單純的 lstm training code.

## 目錄結構

| Name | Description |
| ---- | -------- |
| train.csv | symbolic link to aiaDDD training set |
| yawn_train.csv | symbolic link to YawDD training set |
| test.csv | symbolic link to aiaDDD test set |
| lstm.py | LSTM training code |
| res_mobilecus_512_test | 結合 YawDD/aiaDDD 的 training 結果, 在 aiaDDD test 上每個影片的表現 |

## 執行
```
$ python lstm.py
```
