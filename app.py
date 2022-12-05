import numpy as np
import streamlit as st
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

from PIL import Image

model = keras.models.load_model("./simple31.h5")
def welcome():
	return 'welcome all'


def prediction(image):
    #file = StringIO(image.getvalue().decode("utf-8"))
    #file=image.getvalue()
    #file = load_img(image, target_size=(800,1200))
    file=Image.open(image)
    file=file.resize((80,80))
    #input_arr = np.array(file)
    #print(input_arr.shape)
    #input_arr = input_arr.astype('float32') / 255
    input_arr = img_to_array(file)
    input_arr = np.array([input_arr])
    input_arr = input_arr.astype('int') / 255
    k=model.predict(input_arr)
    return k[0],file

# this is the main function in which we define our webpage
# this is the main function in which we define our webpage
def main():
    
	# giving the webpage a title
	st.title("Imagno")
	
	# here we define some of the front end elements of the web page like
	# the font and background color, the padding and the text to be displayed
	html_temp = """
	<div style ="background-color:yellow;padding:13px">
	<h1 style ="color:black;text-align:center;">Streamlit image quality ML App </h1>
	</div>
	"""
	
	# this line allows us to display the front end aspects we have
	# defined in the above code
	st.markdown(html_temp, unsafe_allow_html = True)
	
	# the following lines create text boxes in which the user can enter
	# the data required to make the prediction
	img_input = st.file_uploader("upload image",type=["png","jpg","jpeg"])
	
	# the below line ensures that when the button called 'Predict' is clicked,
	# the prediction function defined above is called to make the prediction
	# and store it in the variable result
	if st.button("Infer"):
		result,img = prediction(img_input)
		st.image(img,width=500)
		st.image(result,clamp=True,width=500)
	
if __name__=='__main__':
	main()