# YawDD 相關實驗

+ [目錄及檔案](#目錄及檔案)

+ [實驗流程](#實驗流程)

# 目錄及檔案

| Name | Description |
| ---- | -------- |
| mtcnn_face | 使用 MTCNN 找臉進行後續不同 model 的抽 feature 動作, 抽出之 feature 也放在這個目錄中. |
| ssd_face | 使用 SSD 找臉後進行同上的動作. |
| yawn_train.csv | 在 YawDD 中的呵欠影片中選出 74 個當 training video. |
| yawn_valid.csv | 在 YawDD 中的呵欠影片中選出 20 個當 validation video (影響 training 時是否要把目前的 model 存下來). |
| yawn_test.csv | 在 YawDD 中的呵欠影片中選出 24 個當 test video. 作為 Model 最後的評比標準. |
| noyawn_train.csv | 選出 132 個 YawDD 中沒有打呵欠的影片, 在過程中因為效果不好所以加上多一些的 training data. |


# 實驗流程

. 首先先實驗找到人臉之後, 直接抽整張人臉的 feature, 然後送給 LSTM 做 prediction, 但結果不好.
. 認為可能是因為 training data 不夠, 所以增加 noyawn_train.csv 來增加 training data. 但因為增加的都是沒有打呵欠的影片, 所以效果也差不多.
. 想過或許可以視為迴歸問題, 以單張的 feature 來丟入 regression model 來 training, 結果更差.
. 因此開始決定先框出嘴巴區域(根據 SSD face detection 的人臉框取固定比例), 抽完 feature 後再 training, 開始有效果.
. 因為後續是套用 LSTM 類型的模型來預測呵欠程度, 所以也實驗決定抽多少 features 比較好.
. MTCNN face detection 可以偵測出左嘴角、右嘴角的特徵點, 以此來框出嘴巴區域會比固定嘴巴位置更為準確, 結果也是這樣.

