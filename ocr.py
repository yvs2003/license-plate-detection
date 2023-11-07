import streamlit as st
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import cv2

import streamlit as st
from PIL import Image,ImageOps
import numpy as np

from keras.models import load_model

b_0 = np.full((24,1),0)
b_1 = np.full((24,1),1)
b_2 = np.full((24,1),2)
b_3 = np.full((24,1),3)
b_4 = np.full((24,1),4)
b_5 = np.full((24,1),5)
b_6 = np.full((24,1),6)
b_7 = np.full((24,1),7)
b_8 = np.full((24,1),8)
b_9 = np.full((24,1),9)
b_a = np.full((24,1),'a')
b_b = np.full((24,1),'b')
b_c = np.full((24,1),'c')
b_d = np.full((24,1),'d')
b_e = np.full((24,1),'e')
b_f = np.full((24,1),'f')
b_g = np.full((24,1),'g')
b_h = np.full((24,1),'h')
b_i = np.full((24,1),'i')
b_j = np.full((24,1),'j')
b_k = np.full((24,1),'k')
b_l = np.full((24,1),'l')
b_m = np.full((24,1),'m')
b_n = np.full((24,1),'n')
b_o = np.full((24,1),'o')
b_p = np.full((24,1),'p')
b_q = np.full((24,1),'q')
b_r = np.full((24,1),'r')
b_s = np.full((24,1),'s')
b_t = np.full((24,1),'t')
b_u = np.full((24,1),'u')
b_v = np.full((24,1),'v')
b_w = np.full((24,1),'w')
b_x = np.full((24,1),'x')
b_y = np.full((24,1),'y')
b_z = np.full((24,1),'z')
y = np.concatenate([b_0,b_1,b_2,b_3,b_4,b_5,b_6,b_7,b_8,b_9,b_a,b_b,b_c,b_d,b_e,b_f,b_g,b_h,b_i,b_j,b_k,b_l,b_m,b_n,b_o,b_p,b_q,b_r,b_s,b_t,b_u,b_v,b_w,b_x,b_y,b_z])
from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
y = lbl.fit_transform(y)





model1 = load_model("C:\\Users\\User-1\\Downloads\\ocr.h5")
number_plate = cv2.CascadeClassifier('C:\\Users\\User-1\\Desktop\\college\\projects\\License Plate Detection\\haarcascades\\indian_license_plate.xml')

def detect_plate(img, text=''): # the function detects and perfors blurring on the number plate.
    plate_img = img.copy()
    roi = img.copy()
    plate_rect = number_plate.detectMultiScale(plate_img, scaleFactor = 1.2, minNeighbors = 7) # detects numberplates and returns the coordinates and dimensions of detected license plate's contours.
    for (x,y,w,h) in plate_rect:
        roi_ = roi[y:y+h, x:x+w, :] # extracting the Region of Interest of license plate for blurring.
        plate = roi[y:y+h, x:x+w, :]
        cv2.rectangle(plate_img, (x+2,y), (x+w-3, y+h-5), (51,181,155), 3) # finally representing the detected contours by drawing rectangles around the edges.
    if text!='':
        plate_img = cv2.putText(plate_img, text, (x-w//2,y-h//2), 
                                cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.5, (51,181,155), 1, cv2.LINE_AA)
        
    return plate_img, plate # returning the processed image.
    
def find_contours(dimensions, img) :

    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]

    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    
    ii = cv2.imread('contour.jpg')
    
    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs :

        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX) 

            char_copy = np.zeros((36,36))
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (32, 32))
            
            cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
           # plt.imshow(ii, cmap='gray')

            char = cv2.subtract(255, char)

            char_copy[2:34, 2:34] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[34:36, :] = 0
            char_copy[:, 34:36] = 0

            img_res.append(char_copy) 
            
    #plt.show()

    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])
    img_res = np.array(img_res_copy)

    return img_res
    
def segment_characters(image) :

    # Preprocess cropped license plate image
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/6,
                       LP_WIDTH/2,
                       LP_HEIGHT/10,
                       2*LP_HEIGHT/3]
    #plt.imshow(img_binary_lp, cmap='gray')
    #plt.show()
    cv2.imwrite('contour.jpg',img_binary_lp)

    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_binary_lp)

    return char_list

def teachable_machine_classification(img):
    img , number = detect_plate(img)
    char = segment_characters(number)
    df = np.array(char)
    df = df.reshape(-1,36,36,1)
    res = np.argmax(model1.predict(df) , axis=-1)
    a = lbl.inverse_transform(res)
    return a


st.title('Whats your  mood')

uploaded_file = st.file_uploader("Upload an Image...", type=["jpg","png" , "jpeg"])
if uploaded_file is not None:
    image=Image.open(uploaded_file)
    image1=image
    image = np.asarray(image)
    #img = cv2.resize(image , (224,224))
    #img = img.reshape(-1 , 224 , 224 , 3)
    var = str(image.size)
    st.write(var)
    label=teachable_machine_classification(image)
    st.write(label)
   
    
    st.image(image1)
    st.button('Press')