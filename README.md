# yolov5_tensorrtx
1. git clone https://github.com/everguard-inc/yolov5_tensorrtx.git in eg/vsc-pipeline.
2. cd yolov5_tensorrtx/yolov5
3. 
   ```
   mkdir build
   cd build
   ```
   download best weights wts format with 
   ```
   aws s3 cp s3://training-results/ppe_multilabel/yolo_v5/best_adam.wts .
   ```

4. in build folder:
    ```
    cmake ..
    ```
    ```
    make
    ```
    ```
    ./yolov5 -s best_adam.wts best_adam.trt m 
    ```

Finally you should get best_adam.trt in build folder.

5. download images for test inference:
    ```
    mkdir coco_calib
    ```
    ```
    aws s3 sync s3://eg-ukraine-team/rodion/coco_calib/ coco_calib
    ```

6. To run inference on images:
  ```
  python3 yolov5_trt.py
  ```
In yolov5_trt.py you already have trt-model path, path to folder with images:
  ```
  PLUGIN_LIBRARY = "/app/yolov5_tensorrtx/yolov5/build/libmyplugins.so"
  engine_file_path = "build/best_adam.trt"
  image_dir = "build/coco_calib/"
  ```

If you get an error of missing pycuda:
 ```
 wget https://files.pythonhosted.org/packages/5a/56/4682a5118a234d15aa1c8768a528aac4858c7b04d2674e18d586d3dfda04/pycuda-2021.1.tar.gz
 tar xfz pycuda-2021.1.tar.gz
 cd pycuda-2021.1
 python configure.py --cuda-root=/usr/local/cuda (your path to cuda)
 su -c "make install"
 ```

P.S.\
Inference was tested on kyiv-jetson and in eg/vsc-pipeline docker
   
