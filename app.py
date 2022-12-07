import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image
import xgboost
from sklearn.pipeline import Pipeline


# loading in the model to predict on the data
#classifier = pickle.load(open('D:\Mini Project\Political Spectrum\model.pkl', 'rb'))
#C:\Users\Administrator\Desktop\Mini Project\Political Spectrum\model.pkl
#classifier = joblib.load(open(r'D:\Mini Project\Political Spectrum\model.pkl', 'rb'))
classifier = pickle.load(open('pipe.pkl', 'rb'))
def welcome():
	return 'welcome all'

# defining the function which will make the prediction using
# the data which the user inputs
def prediction(text):
	
	prediction = classifier.predict(
		[text])
	print(prediction)
	return prediction
	

# this is the main function in which we define our webpage

def main():
	# giving the webpage a title
	st.title("Poltical Spectrum Classifier")
	
	# here we define some of the front end elements of the web page like
	# the font and background color, the padding and the text to be displayed
	html_temp = """
	<div style ="background-color:yellow;padding:13px">
	<h1 style ="color:black;text-align:center;">Poltical Spectrum Classifier ML App </h1>
	</div>
	"""
	
	# this line allows us to display the front end aspects we have
	# defined in the above code
	st.markdown(html_temp, unsafe_allow_html = True)
	
	# the following lines create text boxes in which the user can enter
	# the data required to make the prediction
	text = st.text_input("Type Here")
	
	result =""
	
	# the below line ensures that when the button called 'Predict' is clicked,
	# the prediction function defined above is called to make the prediction
	# and store it in the variable result
	if st.button("Predict") & len(text)>0:
		result = prediction(text)
	
	if result == 0:
		result = ' Neutral'
	elif result == 1:
		result = ' Left'
	elif result == 2:
		result = ' Right'
	
	st.success('The output is {}'.format(result))
	
if __name__=='__main__':
	main()
