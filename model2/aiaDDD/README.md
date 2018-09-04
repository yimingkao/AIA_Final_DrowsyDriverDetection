# aiaDDD 相關實驗

# 目錄及檔案

| Name | Description |
| ---- | -------- |
| mtcnn_face | 使用 MTCNN 找臉進行後續不同 model 的抽 feature 動作, 抽出之 feature 也放在這個目錄中. |
| ssd_face | 使用 SSD 找臉後進行同上的動作. |
| train.csv | dataset 中 29 個用作 training 的檔案名. |
| test.csv | dataset 中 21 個用作 test 的檔案名. |


# 實驗重點

- 針對 aiaDDD 再次確認 dense121 及 mobilenet 以及做完 transfer learning 的 mobilenet weighting, 比較它們抽取 feature 最後的效果.
- 修改 customized CNN 來加速 mobilenet 的抽取速度.

