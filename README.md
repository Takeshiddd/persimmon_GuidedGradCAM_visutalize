# persimmon_GuidedGradCAM_visutalize

This is a Keras implementation of the paper [論文]. 
This code is for visualization of GuidedGradCAM with persimmon images. 

# Code explanation
## ・calc_ggp_weight.py
To calculate weight data of GuidedGradCAM from input images. 
Put the images of positive labels in ```test/posi``` and the images of negative labels in ```test/nega``` before excute this code. 
In addition, specify the model path at the top of the code as follows:

```
model_path = "./models/vgg16/Thres1_sgd0.001_epoch30_batch32.h5"
```

Weight data will be exported into ```raw_data```.

## ・GCC_data_format.py
To format the weight data exported by calc_ggp_weight.py for visualization. 

After excute this code, the formated data will saved in ```GuidedGradCAM_data```.

## ・export_heatmap.py
To export heatmaps on top of images.
Before excute this code, put the all input images in ```GGC-pictures```.
The output images will be saved in ```GuidedGradCAM_visualization_with_image```.

## ・get_hist.py
To export histogram of wights.
The export dir is ```histgram```.