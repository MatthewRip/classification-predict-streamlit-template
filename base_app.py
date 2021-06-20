"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
from typing import Text
from altair.vegalite.v4.schema.core import Categorical
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import numpy as np

# Vectorizer
#news_vectorizer = open("resources/tfidfvect.pkl","rb")
news_vectorizer = open("./tfidf_vectorizer.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Climate Change Analysis With Tweet Classifer")
	st.subheader("Climate change tweet classification")
	st.write('This is a web app to predict the text category based on\
        the tweet text about climate change. Please enter the\
        text that includes climate change. After that, click on the Classify button at the bottom to\
        see the prediction of the classifier.')

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information", "Team_members", "EDA", "Overview"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Information of our Twitter raw data and Labels. Tick the box below to view the raw data")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

		if not st.checkbox('view Graph', False, key=1):
			st.subheader("Comparison of people who belief in climate change")
			sent =pd.DataFrame(raw['sentiment'].value_counts())
			st.bar_chart(sent)
			st.line_chart(sent)
			

	#Building out the cover page
	if selection == "Overview":
		st.info("**What is Climate Change**")
		st.image("https://res.cloudinary.com/limoncloud/image/upload/v1623437693/climate-change-2254711_1920_mvmcum.jpg")
		st.markdown("Climate change is a change in the usual weather found in a place.\nIt could be a change in how much rain a place usually get in a year/months/season.\nIt is also a change in Earth's Climate")
		st.subheader("**What Is Causing Earth's Climate to change?**")
		st.markdown("Many things can cause climate change. **For Example**: Oceans can change, Earth distance\nfrom the sun can change and Volcano eruptions can change our climate. Also human can change\nclimate too. One way to get energy is to burn Coals,Oils and Gas. Burning those can cause\nthe air to head up,this can change climate of a place and change Earth's climate.")
		st.subheader("**What Might Happen To Earth's Climate?**")
		st.markdown("Scientist say Earth's temperature will keep going up,and this will cause more snow and ice\nto melt. Oceans will rise higher,some places might get more rain and others less and other\nplaces might have stronger Hurricanes.")
		st.subheader("**Better Solutions To Climate**")
		st.markdown("Planting more trees 🏞️🏝️🏜️\nSave Energy 🔋🔌\nUse Water wisely 💦🚿💧.")

	#Building out the EDA page
	if selection == "EDA":
		st.header("Exploratory data analysis")
		st.subheader("Tweet Dataset")
		if not st.checkbox('show Data', False, key=1):
			st.text("showing data")
			st.write(raw[['sentiment', 'message']])
			feature_labels = st.radio("select to view", ("columns", "rows"))
		if st.button("About us"):
			st.text("Hello World")


	# #Building out the Team Members page
	# if selection == "Team_members":
	# 	st.info("**Our Team:**")
	# 	col1, col2 = st.beta_columns(2)
	# 	with col1:
	# 		st.header("Lerato Mohlala")
	# 		st.image("https://res.cloudinary.com/limoncloud/image/upload/v1623430480/IMG_20210611_185350_023_fko9ia.jpg",caption= "EDSA-Student", width=100)
	# 	with col2:
	# 		st.header("Matthew Rip")
	# 		st.image("https://res.cloudinary.com/limoncloud/image/upload/v1623430480/IMG_20210611_185350_023_fko9ia.jpg",caption= "EDSA-Student", width=100)

	# 	col3, col4 = st.beta_columns(2)
	# 	with col3:
	# 		st.header("Mukondeleli Negukhula")
	# 		st.image("https://res.cloudinary.com/limoncloud/image/upload/v1623430480/IMG_20210611_185350_023_fko9ia.jpg",caption= "EDSA-Student", width=100)
	# 	with col4:
	# 		st.header("Rejoice Van der Walt")
	# 		st.image("https://res.cloudinary.com/limoncloud/image/upload/v1623430480/IMG_20210611_185350_023_fko9ia.jpg",caption= "EDSA-Student", width=100)
	
	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text",'prediction text')

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			#predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			predictor = joblib.load(open(os.path.join("./model.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

			st.subheader('Prediction')
			text_Category = np.array(['-1','0','1','2'])
			st.write(text_Category[prediction[:]])
		
		

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
