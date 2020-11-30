Textured SMPL

### Build textured 3D body model with two images
example images

## Requirement
* numpy
* scipy
* OpenCV
* [smplify-x](https://github.com/vchoutas/smplify-x)

## Optional requirement

* [PGN](https://github.com/Engineering-Course/CIHP_PGN)

## Demo
We have an exmple data in `data/obj1`. The front iamge is data/obj1/images/P01125-150055.jpg and the back image is data/obj1/images/P01125-150146.jpg.

To generate the texture for SMPL model:  `python demo.py data/obj1 P01125-150055.jpg P01125-150146.jpg`
To generate the texture for SMPLX model:  `python demo.py data/obj1 P01125-150055.jpg P01125-150146.jpg`

You'll find the results in `data/obj1`


## Using your only data
### step1: prepare your image data
example can be find in `./data/obj1/images`

### step2: openpose pose detection
For Linux users, you need to complie openpose following the instruction here, and for windows users, you can used the prebuild windows demo.
`openpose.bin --display 0  --render_pose 1 --image_dir ./data/obj1/images --write_json ./data/obj1/keypoints --write_images ./data/obj1/pose_images --hand --face`

### step3: fit smpl/smplx model
Please follow the instruction [here](https://github.com/vchoutas/smplify-x)
An example command is:
`python smplifyx/main.py --config cfg_files/fit_smplx.yaml --data_folder ./data/obj1 --output_folder ./data/obj1/smplx --visualize=True  --model_folder models --vposer_ckpt vposer_v1_0`
Please copy the output data to ./data/obj1/smplx or ./data/obj1/smpl

### step4(optional): get PGN segmentation
Please follow the instruction [here](https://github.com/Engineering-Course/CIHP_PGN)
Please copy the output data to ./data/obj1/PGN

### step4: texture generation
run `python demo.py data_path front_img back_img smplx`



