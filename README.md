# AIA_Final_DrowsyDriverDetection
人工智慧學校的期末專題 - 疲勞駕駛偵測

### Dataset 位置
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
jovyan@jupyter-at072xxx:at072-group4$ ./gdown.pl https://drive.google.com/file/d/1xMBUe0hUweyTb5Hz5WTxSNSaDZX07yxk mark.zip
```
(建議在 hub 上使用 tree mode 而非 lab mode, 這樣子 terminal 才可以使用貼上功能.)
抓完後 unzip 到別的資料夾中, 再 copy 裡面的 csv 檔到 aiaDDD 這個 folder 中.
```
jovyan@jupyter-at072xxx:at072-group4$ ./gdown.pl https://drive.google.com/file/d/1xMBUe0hUweyTb5Hz5WTxSNSaDZX07yxk mark.zip
jovyan@jupyter-at072xxx:at072-group4$ unzip mark.zip          (會產生 mark 資料夾)
jovyan@jupyter-at072xxx:at072-group4$ mv mark/*.csv aiaDDD/   (將 mark 檔放到 dataset 中)
jovyan@jupyter-at072xxx:at072-group4$ rm -rf mark mark.zip    (清掉暫存檔. mark 資料夾中有將 mark 的結果視覺化, 可看標的準不準)
```

