# AIA_Final_DrowsyDriverDetection
人工智慧學校的期末專題 - 疲勞駕駛偵測

+ [目錄結構](#目錄結構)

+ [流程](#流程)


# 目錄結構
根目錄這邊有資料集、人臉偵測、特徵抽取、以及模型四個子目錄.
```
+ YawDD              (資料集 - 標記檔)
+ aiaDDD             (資料集 - 標記檔)
+ SSDface            (使用 SSD 做人臉偵測)
+ MTCNNface          (使用 MTCNN 做人臉偵測, 同時找人臉的五個關鍵點)
+ feaExtract         (使用 dense121, mobilenet, mobilenet_v2 來抽取 features)
+ model1             (單張不做人臉偵測直接偵測呵欠程度 - Failed)
+ model2             (多張、人臉偵測、特徵抽取、模型判斷呵欠程度 - Good)
```
        
# 流程
![系統流程及使用的模型](flow.png)