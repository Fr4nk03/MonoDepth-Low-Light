# MonoDepth-Low-Light

You can predict scaled disparity for a single image or a folder of images with:
```
python test_simple_batch.py --image_path ../test/images --model_name mono+stereo_640x192
```

Enhance the estimated depth map with Gaussian-Sobel Filtering:
```
python gaussian_sobel_simple.py --image_path ../test/output --output_path ../test/output_enhanced
```