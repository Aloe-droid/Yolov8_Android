# Yolov8_Android

## Yolo V8 nano 화재 검출 모델입니다. 

------------

#### 학습은 Yolo V8 공식사이트(https://github.com/ultralytics/ultralytics) 를 사용했습니다.
  * 학습 방식입니다. (공식 사이트에도 있습니다.) python 코드입니다.
    * cmd 창을 킵니다.
    * pip install ultralytics 
    * (https://universe.roboflow.com/) 해당 사이트에서 학습하고싶은 데이터를 v8 버전으로 다운받습니다.
    * 해당 폴더에 아래와 같은 python 파일을 생성합니다. ("fire.yaml" -> 다운받은 yaml 파일의 이름으로 변경하시면 됩니다.)
```
from ultralytics import YOLO
from multiprocessing import freeze_support

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

if __name__ == '__main__':
    freeze_support()  # for Windows support

    model.train(data="fire.yaml", epochs=100)  # train the model
```

  * 다음은 onnx 변환 파일입니다. 같은 폴더에 아래와 같은 python 파일을 생성합니다. (위의 학습 코드와 합치셔도 무관합니다.)
  
```
from ultralytics import YOLO

model = YOLO("절대 경로를 적어주시면 됩니다\\best.pt", type="v8")
model.fuse()  
model.info(verbose=True)  # Print model information
model.export(format= "onnx")  # TODO:  export to ONNX format
``` 

------------

#### 해당 앱은 가로 모드로 고정되어있습니다. 가로모드로 사용해주세요.
#### yolo v5는 실행되지 않습니다. 다른 깃허브를 참고해주시면 됩니다.
