# AIA_Final_DrowsyDriverDetection
人工智慧學校的期末專題 - 疲勞駕駛偵測

+ [資料集](#Dataset)

+ [目錄結構](#目錄結構)

+ [流程](#流程)

+ [嘗試與結果](#嘗試與結果)

## Dataset
原先 yawDD 的位置, 助教已經幫忙將檔案放到 server 上 '/projectdata/driver' 上
```
jovyan@jupyter-at072xxx:~$ cd /projectdata/driver
jovyan@jupyter-at072xxx:/projectdata/driver$ ls
opencv_dlib_fd opencv_dnn_fd openface_landmark YawDD 台灣人工智慧學校_疲勞駕駛偵測.docx
```
其中除了 YawDD 資料夾之外, 其它都是利用程式標記過的人臉資訊, 詳細的 format 及內容可以看 台灣人工智慧學校_疲勞駕駛偵測.docx.


目前開起 server 後, 可以看到多一個 folder,
```
jovyan@jupyter-at072xxx:~$ ls
* at072-group04 courses projectdata
```
這個 'at072-group04' (或是 05) 就是每個 group 中共用的資料 folder, 大家可以在裡面跑程式並且分享檔案.
裡面已經放了我們的 dataset -- aiaDDD.
這邊沒有 mark 的資訊, 所以要從 github 上面抓下來.
```
jovyan@jupyter-at072xxx:at072-group04$ ./gdown.pl https://drive.google.com/file/d/1xMBUe0hUweyTb5Hz5WTxSNSaDZX07yxk mark.zip
```
(建議在 hub 上使用 tree mode 而非 lab mode, 這樣子 terminal 才可以使用貼上功能.)
抓完後 unzip 到別的資料夾中, 再 copy 裡面的 csv 檔到 aiaDDD 這個 folder 中.
```
jovyan@jupyter-at072xxx:at072-group04$ ./gdown.pl https://drive.google.com/file/d/1xMBUe0hUweyTb5Hz5WTxSNSaDZX07yxk mark.zip
jovyan@jupyter-at072xxx:at072-group04$ unzip mark.zip          (會產生 mark 資料夾)
jovyan@jupyter-at072xxx:at072-group04$ mv mark/*.csv aiaDDD/   (將 mark 檔放到 dataset 中)
jovyan@jupyter-at072xxx:at072-group04$ rm -rf mark mark.zip    (清掉暫存檔. mark 資料夾中有將 mark 的結果視覺化, 可看標的準不準)
```

# 目錄結構
下面是指 at072-group04 下面的目錄結構.
```
+ at072-group04
    + aiaDDD             (資料集)
    + model1             (first try, image -> nn prediction, failed, 可以不管它)
    + model2             (第二個嘗試的作法, image -> face bbox -> feature extraction -> nn prediction)
        + aiaDDD
        + YawDD
            + ssd_face   (使用 ssd face detection 做出的人臉位置, 後續相關實驗)
                + bbox   (人臉框的資訊)
                + full   (使用完整人臉來抽 features)
                + roi    (使用選定的 roi 來抽 features)
```
        
# 流程
我們的處理 pipeline 是
  - 前處理
  - 找人臉位置
  - 抽 feature
  - Training or Inference
  
# 嘗試與結果
  - model1: 直接整張圖抽 features 再 training & inference.
