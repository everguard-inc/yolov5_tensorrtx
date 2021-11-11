# yolov5_tensorrtx
1. git clone https://github.com/everguard-inc/yolov5_tensorrtx.git in eg/vsc-pipeline.
2. cd yolov5_tensorrtx/yolov5
3. 
   ```
   mkdir build
   cd build
   ```
   download best weights wts format for 10 classes: 
   * in_harness
   * not_in_harness 
   * harness_unrecognized
   * in_vest
   * not_in_vest 
   * vest_unrecognized
   * in_hardhat
   * not_in_hardhat
   * hardhat_unrecognized
   * crane_bucket
   ```
   aws s3 cp s3://eg-ukraine-team/rodion/crane_buckets/cl10_bucket.wts .
   ```

4. in build folder:
    ```
    cmake ..
    ```
    ```
    make
    ```
    ```
    ./yolov5 -s cl10_bucket.wts cl10_bucket.trt m 
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
  python3 inference.py
  ```
In yolov5_trt.py you already have trt-model path, path to folder with videos:
  ```
  PLUGIN_LIBRARY = "/app/yolov5_tensorrtx/yolov5/build/libmyplugins.so"
  engine_file_path = "build/cl10_bucket.trt"
  video_path = "test_videos/"
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
   
