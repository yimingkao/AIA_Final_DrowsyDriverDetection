# YawDD - SSD Face Detect 相關實驗


## 目錄結構

| Name | Description |
| ---- | -------- |
| bbox | 每個影片使用 SSD 找臉得到的 bounding box 位置. |
| full | 利用整個人臉區域來作相關的實驗. |
| roi | 手動標出人臉嘴巴區域並作相關的實驗. |
| ssd_face_det.py | SSD 找人臉的 class code. |
| ssdfd2bbox.py | 利用 ssd_face_det.py 來找每個影片中人臉的位置並存至 bbox 資料夾中. |
| verify.py | 將找到的人臉位置畫在 video 上面並存下, 檢視找到人臉位置的正確性. |

## 操作順序

```
$ python ssdfd2bbox.py  => 存入 bbox 目錄中
$ python verify.py => 將存入 bbox 目錄中的座標畫在 video 上做驗證.
```

