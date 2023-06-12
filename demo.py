import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from scipy import sparse
import anndata
from keras.models import load_model, Model
from tensorflow.keras.utils import to_categorical
from keras import backend as K
import cv2
import os
from mtcnn import MTCNN
import gdown

def remove_sparsity(adata):
    if sparse.issparse(adata.X):
        adata.X = adata.X.A
    return adata

conditions = ['r', 'c', 'a']
labelencoder = {'r': 0, 'c': 1, 'a': 2}
input_shape = (64, 64, 3)

class DCtrVAE:
    def __init__(self, x_dimension, **kwargs):
        self.x_dim = x_dimension if isinstance(x_dimension, tuple) else (x_dimension,)
        self.n_conditions = kwargs.get("n_conditions", 2)
        self.model_path = kwargs.get("model_path", "./")

    def predict(self, adata, encoder_labels, decoder_labels):
        adata = remove_sparsity(adata)

        images = np.reshape(adata.X, (-1, *self.x_dim))
        encoder_labels = to_categorical(encoder_labels, num_classes=self.n_conditions)
        decoder_labels = to_categorical(decoder_labels, num_classes=self.n_conditions)

        reconstructed = self.cvae_model.predict([images, encoder_labels, decoder_labels])[0]
        reconstructed = np.reshape(reconstructed, (-1, np.prod(self.x_dim)))

        reconstructed_adata = anndata.AnnData(X=reconstructed)
        reconstructed_adata.obs = adata.obs.copy(deep=True)
        reconstructed_adata.var_names = adata.var_names

        return reconstructed_adata

    def restore_model(self):
        self.cvae_model = load_model(os.path.join(self.model_path, 'mmd_cvae.h5'), compile=False, custom_objects={'Functional':Model})

class FaceDetector(object):
    def __init__(self, xml_path):
        print(xml_path)
        self.classifier = cv2.CascadeClassifier(xml_path)
    
    def detect(self, image, biggest_only=True):
        scale_factor = 1.2
        min_neighbors = 5
        min_size = (30, 30)
        biggest_only = True
        faces_coord = self.classifier.detectMultiScale(image,
                                                       scaleFactor=scale_factor,
                                                       minNeighbors=min_neighbors,
                                                       minSize=min_size,
                                                       flags=cv2.CASCADE_SCALE_IMAGE)
        return faces_coord
def cut_faces(image, faces_coord):
    faces = []
    
    for (x, y, w, h) in faces_coord:
        w_rm = int(0.3 * w / 2)
        faces.append(image[y: y + h, x + w_rm: x + w - w_rm])
         
    return faces


# Define a function to preprocess the image and make a prediction using the loaded model
def predict(image, source, target, network, detector2, crop=True):
    if crop:
        #pil_image = Image.open(image)
        #pil_num = np.array(pil_image)
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        temp_num_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #temp_num_im = cv2.cvtColor(pil_num, cv2.COLOR_BGR2RGB)
        if source == 'a' or source == 'd':
            faces_coord = [det['box'] for det in detector2.detect_faces(temp_num_im)]
        else:
            xml = "./haarcascade_frontalface_default.xml"
            detector = FaceDetector(xml)
            faces_coord = detector.detect(temp_num_im, True)
        if len(faces_coord) != 0:
            print("Braaaa")
            face = cut_faces(temp_num_im ,faces_coord)[-1]
            im_o = Image.fromarray(face.astype('uint8'), 'RGB')
        else:
            print("Broooo")
            im_o = Image.open(image)
    else:
        im_o = Image.open(image)

    im = ImageOps.exif_transpose(im_o)
    im = im.convert("RGB").resize((64,64))
    num_im = np.array(im)
    num_im = num_im.astype('float32').reshape(-1,)
    source_adata = anndata.AnnData(X=np.asarray([num_im]))
    source_adata = remove_sparsity(source_adata)

    source_adata.X /= 255.0

    source_labels = np.zeros(source_adata.shape[0]) + labelencoder[source]
    target_labels = np.zeros(source_adata.shape[0]) + labelencoder[target]

    pred_adata = network.predict(source_adata,
                                encoder_labels=source_labels,
                                decoder_labels=target_labels,
                                )
    arr2 = pred_adata[pred_adata.obs.index == "0"].X
    arr2 = (arr2-np.min(arr2))/(np.max(arr2)-np.min(arr2))
    arr2 = np.reshape((arr2*255), (64, 64, 3))
    img2 = Image.fromarray(arr2.astype('uint8'), 'RGB')

    img_up = img2.resize((512, 512))

    col1, col2 = st.columns(2)
    with col1:
        st.image(im_o, use_column_width=True)
    with col2:
        st.image(img_up, use_column_width=True)

label_map = {"real": 'r', "anime": 'a'}

@st.cache_resource
def load_model1():
    # Doanload pretrained model
    #url = 'https://drive.google.com/uc?id=1nVvW4f77yomKqOV0-Hf6QUc_xZb6CssR'
    #output = './anime_cartoon1b-a/mmd_cvae.h5'
    #gdown.download(url, output, quiet=False)

    # Load the model
    network = DCtrVAE(x_dimension=input_shape,
                                      n_conditions=len(conditions),
                                      model_path=f"./anime_cartoon1b-a")
    network.restore_model()
    print("Model 1 Restored")
    return network

@st.cache_resource
def load_detector2():
    detector2 = MTCNN()
    return detector2

def load_image():
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
    col1, col2 = st.columns([4, 1])
    with col1:
        label = st.selectbox('What type of image is this?', ('real', 'anime'))
    with col2:
        crop = st.checkbox('Cropped?')
    return uploaded_file, label, crop

def load_examples():
    st.write("Here are some example images to try:")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image("./eilish.png", use_column_width=True)
    with col2:
        st.image("./tom.png", use_column_width=True)
    with col3:
        st.image("./shikamaru.png", use_column_width=True)
    with col4:
        st.image("./hinata.jpg", use_column_width=True)

def main():
    st.title('Anime TrVAE')
    model1 = load_model1()
    #model2 = load_model2()
    detector2 = load_detector2()
    image, label, crop = load_image()
    label = label_map[label]
    to_label = 'r' if label == 'a' else 'a'
    if st.button('Predict'):
        predict(image, label, to_label, model1, detector2, crop=crop)

if __name__ == '__main__':
    main()
