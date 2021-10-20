# yolov5_tensorrtx
1. git clone https://github.com/everguard-inc/yolov5_tensorrtx.git in eg/vsc-pipeline.
2. cd yolov5_tensorrtx  
   Then git clone https://github.com/everguard-inc/yolov_5_multilabel.git
3. cd yolov_5_multilabel and download best weights with 
   ```
   aws s3 cp s3://training-results/ppe_multilabel/yolo_v5/best_adam.pt .
   ```
4. Create wts file with 
   ```
   python3 gen_wts.py -w best_adam.pt -o best_adam.wts
   ```
5. ```cd yolov5_tensorrtx/yolov5/ \
   mkdir build
   ```
   move wts file to build folder with 
   ```
   cp yolov5_tensorrtx/yolov_5_multilabel/best_adam.wts yolov5_tensorrtx/yolov5/build/
   ```
6. download images for test inference:
    ```
    cd build
    ```
    ```
    mkdir coco_calib
    ```
    ```
    aws s3 sync s3://eg-ukraine-team/rodion/coco_calib/ coco_calib
    ```
7. in build folder:
    ```
    cmake ..
    ```
    ```
    make
    ```
    ```
    ./yolov5 -s best_adam.wts best_adam.trt m 
    ```

Finally you should get best_adam.trt in build folder.\

8. To run inference on images:
  ```
  python3 yolov5_trt.py
  ```
In yolov5_trt.py you already have trt-model path, path to folder with images:
  ```
  engine_file_path = "build/best_adam.trt"
  image_dir = "build/coco_calib/"
  ```
Inference was tested on kyiv-jetson and in eg/vsc-pipeline docker
   
