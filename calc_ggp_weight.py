
import matplotlib.pyplot as plt
from keras.models import load_model, Model
import numpy as np
from visualizations import GradCAM, GuidedGradCAM, GBP, LRPB, LRPE
from helper import heatmap
import innvestigate.utils as iutils
import os
from keras.preprocessing.image import img_to_array, load_img
import skimage as sk
sk.__version__

def make_directory(dir_paths):
    for dir_path in dir_paths:
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
            

model_path = "./models/vgg16/Thres1_sgd0.001_epoch30_batch32.h5"
# limits tensorflow to a specific GPU
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# make export dir
raw_data = 'raw_data'
make_directory([raw_data])


model = load_model(model_path) # input VGG trained model
# model.load_weights("models/tensorlog/inceptionV3/weights.36-0.96-0.84.hdf5")
model.layers[-2].name='dense_1_'
model.summary()

# Only the partial model is needed for the visualizers. Use innvestigate.utils.keras.graph.pre_softmax_tensors()
partial_model = Model(
    inputs=model.inputs,
    outputs=iutils.keras.graph.pre_softmax_tensors(model.outputs),
    name=model.name,
)
partial_model.summary()


# Range of input images
# keras_applications VGG16 weights assume a range of (-127.5, 127.5). Change this to a range suitable for your model.
max_input = -127.5
min_input = 127.5
# max_input = 0
# min_input = 256


from tqdm import tqdm
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix


imagesize = 224
width = height = imagesize

def read_pil_image(img_path, height, width):
        with open(img_path, 'rb') as f:
            return np.array(Image.open(f).convert('RGB').resize((width, height)))

prediction = []
label = []
datapath_posi = "./test/posi/"
datapath_nega = "./test/nega/"
files = os.listdir(datapath_posi)
for p in tqdm(files[:]):
    img = read_pil_image(datapath_posi + p, height, width)
    img = img.reshape((1,imagesize,imagesize,3))
    prediction.append(model.predict(img))
    label.append([0,1])
    
files = os.listdir(datapath_nega)
for p in tqdm(files[:]):
    img = read_pil_image(datapath_nega + p, height, width)
    img = img.reshape((1,imagesize,imagesize,3))
    prediction.append(model.predict(img))
    label.append([1,0])
label = np.array(label).reshape((-1,2))
prediction = np.array(prediction).reshape((-1,2))
print(label.shape)
print(prediction.shape)

y_pred_onehot = prediction
y_pred = np.argmax(y_pred_onehot, axis=1)
y_pred_value = [y_pred_onehot[i][1] for i in range(y_pred.shape[0])]

y_val_onehot = label
y_val = np.argmax(y_val_onehot, axis=1)

# y_val =validation_dataflow.classes
# y_val_onehot = np_utils.to_categorical(y_val)

print('Confusion Matrix')
print(confusion_matrix(y_val, y_pred))


# Change this to load a list of images you want. For this example, we are only loading one image, but you can load a list of files.

#os.system("ls ./test/posi/* > list-posi.txt")
list = open("list.txt", 'r') #list with only names, not including the path
GuideBP = open(os.path.join(raw_data, "GBP-raw-data.txt", "w"))
GuideGC = open(os.path.join(raw_data,"GGC-raw-data.txt", "w"))
LRPSB = open(os.path.join(raw_data,"LRPSB-raw-data.txt", "w"))
LRPEp = open(os.path.join(raw_data,"LRPEp-raw-data.txt", "w"))

singleimg = list.readline()

while singleimg:
    singleimg = singleimg.rstrip()
    orig_imgs = [img_to_array(load_img("./test/"+singleimg, target_size=(imagesize, imagesize)))] # indicate the path 

    input_imgs = np.copy(orig_imgs)
    example_id = 0
    target_class = 1

    # GradCAM and GuidedGradCAM requires a specific layer
    target_layer = "block5_conv3" # VGG only
    predictions = model.predict(input_imgs)
    pred_id = np.argmax(predictions[example_id])
    
    use_relu = False
    
    guidedgradcam_analyzer = GGC(
        partial_model,
        target_id=target_class,
        layer_name=target_layer,
        relu=False,
        allow_lambda_layers = True

    )
    analysis_guidedgradcam = guidedgradcam_analyzer.analyze(input_imgs) 
    np.set_printoptions(threshold=1000000)
    print(singleimg + "\n" + str(analysis_guidedgradcam) + "\nend\n", file=GuidedGC)    

    #Guided Back Propagation
    guidedbackprop_analyzer = GBP(
        partial_model,
        target_id=target_class,
        relu=use_relu,
        allow_lambda_layers = True
    )
    analysis_guidedbackprop = guidedbackprop_analyzer.analyze(input_imgs)    
    np.set_printoptions(threshold=1000000)
    print(singleimg + "\n" + str(analysis_guidedbackprop) + "\nend\n", file=GuideBP)    

    
    #LRP Sequential B
    lrpb_analyzer = LRPB(
        partial_model,
        target_id=target_class,
        relu=False,
        allow_lambda_layers = True
    )
    analysis_lrpb = lrpb_analyzer.analyze(input_imgs)
    np.set_printoptions(threshold=1000000)
    print(singleimg + "\n" + str(analysis_lrpb) + "\nend\n", file=LRPSB)    

    
    #LRP Epsilon
    lrpe_analyzer = LRPE(
        partial_model,
        target_id=target_class,
        relu=False,
        allow_lambda_layers = True
    )
    analysis_lrpe = lrpe_analyzer.analyze(input_imgs)
    print(singleimg + "\n" + str(analysis_lrpe) + "\nend\n", file=LRPEp)    

    singleimg = list.readline()
    
    
list.close()
GuideBP.close()
GuideGC.close()
LRPSB.close()
LRPEp.close()
