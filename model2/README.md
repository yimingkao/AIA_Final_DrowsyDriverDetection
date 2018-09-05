# Model2 - 多張人臉偵測、特徵擷取、模型判斷.

+ [目錄結構](#目錄結構)

+ [流程](#流程)

# 目錄結構
子目錄包含了資料集、嘴巴特徵抽取訓練、判斷模型的訓練、demo流程四個子目錄.

| Name | Description |
| ---- | -------- |
| YawDD | Training, Test set, 不同人臉偵測模型下抽取特徵 |
| aiaDDD | Training, Test set, 不同人臉偵測模型下抽取特徵 |
| mouthnn | 使用 SSD 找人臉並取固定比例之位置當成嘴巴區域, training mobilenet feature 抽取的能力 |
| mouthnn_mtcnn | 使用 MTCNN 找人臉並取回傳之特徵點位置, training mobilenet feature 抽取的能力 |
| mouthnn_ym | 使用 MTCNN 找人臉並取回傳之特徵點位置, 嘗試再縮小 mobilenet 為 customized CNN |
| mouthlstm | 經由 mouthnn 所抽出之每張 feature 及呵欠程度, 訓練一個判斷模型 |
| mouthlstm_mtcnn | 同上, 主要使用基於 MTCNN 之結果 |
| mouthlstm_ym | 同上, 主要使用 customized CNN 的結果 |
| flow | 應用 mouthlstm 訓練好之模型, 在 aiaDDD/YawDDD 的測試資料集上跑出模擬結果, 以及及時的 demo 流程 |
| flow_mtcnn | 同上, 主要使用基於 MTCNN 之結果 |


# 流程
專案進行中先試作以 SSD 找臉的方式, 預定義人臉嘴巴位置, 在 YawDD (資料集抽features) -> mouthnn (修改 imagenet weighting 變得更好) -> mouthlstm (得到一個 prediction model) -> flow (驗證資料集中 test set 結果及 realtime demo flow) 

後續又導入了 MTCNN 的人臉偵測, 它同時會給出兩個嘴角的特徵點, 依此去框出嘴巴位置會更為準確, 結果也更好.
