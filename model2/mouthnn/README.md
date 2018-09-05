# Mouthnn - 基於 YawDD 及 aiaDDD 來找出適合的 mobilenet weighting.


## 檔案及目錄

| Name | Description |
| ---- | -------- |
| pic_gen_aiaDDD.py | 每個檔案去統計每個不同程度的 frame 張數, 最後隨機取一樣多的量來 training. |
| pic_gen_YawDD.py | 同上. |
| pic_train.py | 利用產生的圖檔來做後續 training 的動作. |


## 執行順序
```
$ python pic_gen_aiaDDD.py (會產生 deg0 ~ deg5 的 folder)
$ python pic_gen_YawDD.py
$ python pic_train.py (會把 deg0 ~ deg5 裡面的圖 copy 到 train, test folder 下, 最後 train 出來的結果是 mobilecus_fea_512.h5)
```
