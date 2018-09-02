# YawDD - 整張人臉偵測相關實驗


## 目錄結構

| Name | Description |
| ---- | -------- |
| config.py | 設定要選擇的 feature 個數 |
| bbox2fea_full.py | 每個 video 每張 frame 抽取預定義的 feature 數並存成 .npy 檔 |
| dense_fea_extract.py | dense121 抽 feature 的 class code |
| lstm.py | 讀入 .npy 檔, 利用 LSTM 的方式來做訓練 |
| lstm_inference.py | 讀入存入的 LSTM model, 對於 training set / testing set 做推論動作, 得知結果. |
| lstm_noyawn.py | 由於效果不好, 所以加入 YawDD 中沒有打呵欠的影片一起 training |
| single_classify.py | 轉念想是否可以變成迴歸問題, 直接找一些 regression model 來試試. 效果不佳. |
| single_classify_balance.py | 不打呵欠的圖片居多, 所以修改 training set 中每個呵欠程度的的圖片數量, 效果仍然不佳. |
| single_classify_balance_norm.py | 嘗試加入 feature 的 normalization, 效果還是不佳. |

## 操作順序

```
```

