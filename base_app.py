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
from altair.vegalite.v4.schema.channels import Column
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

	#Building out the cover page
	if selection == "Overview":
		st.info("**What is Climate Change?**")
		st.image("https://res.cloudinary.com/limoncloud/image/upload/v1623437693/climate-change-2254711_1920_mvmcum.jpg")
		st.markdown("Climate change is a change in the usual weather found in a place.\
			It could be a change in how much rain a place usually get in a year/months/season.\
				It is also a change in Earth's Climate")
		st.subheader("**What Is Causing Earth's Climate to change?**")
		st.markdown("Many things can cause climate change. **For Example**: Oceans can change, Earth distance\
			from the sun can change and Volcano eruptions can change our climate. Also human can change\
				climate too. One way to get energy is to burn Coals,Oils and Gas. Burning those can cause\
					the air to head up,this can change climate of a place and change Earth's climate.")
		st.subheader("**What Might Happen To Earth's Climate?**")
		st.markdown("Scientist say Earth's temperature will keep going up,and this will cause more snow and ice\
			to melt. Oceans will rise higher,some places might get more rain and others less and other\
				places might have stronger Hurricanes.")
		st.subheader("**Better Solutions To Climate**")
		st.markdown("* Planting more trees :seedling: :ear_of_rice: :deciduous_tree:")
		st.markdown("* Save Energy :battery: :electric_plug:")
		st.markdown("* Use Water wisely 💦🚿💧")

	#Building out the EDA page
	if selection == "EDA":
		st.header("Exploratory data analysis")
		st.subheader("**Tweet Dataset**")
		if st.checkbox('Preview Dataset', False, key=1):
			if st.checkbox("showing data"):
				st.write(raw[['sentiment', 'message']].head())
			st.markdown("* To view all the **Dataset** go to Information page.")
			
			#Show columns names
			if st.checkbox("showing columns names"):
				st.write(raw.columns)
			
			#Showing Dimensions
			feature_labels = st.radio("select to view", ("Rows", "Columns", "All"))
			if feature_labels == "Columns":
				st.write("Total number of columns in the dataset: {}".format(raw.shape[1]))
			elif feature_labels == "Rows":
				st.write("Total number of rows in the dataset: {}".format(raw.shape[0]))
			else:
				st.write("Total shape of dataset: {}".format(raw.shape))

			#Show summary of the data
			if st.checkbox("show Dataset summary"):
				st.write(raw.describe())
				if st.checkbox("Select column"):
					Col_option = st.selectbox("Select column",("sentiment","tweetid","message"))
					if Col_option == "sentiment":
						st.write(raw["sentiment"])
					if Col_option == "tweetid":
						st.write(raw["tweetid"])
					if Col_option == "message":
						st.write(raw["message"])

		#Plotting graphs
		st.subheader("**Graph Plot**")
		if st.checkbox("showing Bar Graph"):
				st.bar_chart(raw['sentiment'].value_counts())
				st.text("-1: 'Negetive' Tweet does not believe in man-made climate change")
				st.text("0: 'Neutral' Neither the tweet supports nor refuses the believe of man-made climate change")
				st.text("1: 'Positive' Tweet supports the believe of climate change")
				st.text("2: 'News' Tweet links to factual news about climate change")
			
		if st.button("About us"):
			st.text("Tweet Classifier App. Build with Streamlit")

	# #Building out the Team Members page
	if selection == "Team_members":
		st.info("**Our Team:**")
		col1, col2 = st.beta_columns(2)
		with col1:
			st.header("Lerato Mohlala")
			st.image("https://res.cloudinary.com/limoncloud/image/upload/v1623430480/IMG_20210611_185350_023_fko9ia.jpg",caption= "EDSA-Student", width=150)
		with col2:
			st.header("Matthew Rip")
			st.image("https://res.cloudinary.com/limoncloud/image/upload/v1624377289/IMG_20210622_175320_507_ss3efp.jpg",caption= "EDSA-Student", width=150)

		col3, col4 = st.beta_columns(2)
		with col3:
			st.header("Mukondeleli Negukhula")
			st.image("https://res.cloudinary.com/limoncloud/image/upload/v1624197385/IMG_20210620_155017_619_tub8j1.jpg",caption= "EDSA-Student", width=190)
		with col4:
			st.header("Rejoice Van der Walt")
			st.image("https://res.cloudinary.com/limoncloud/image/upload/v1623430480/IMG_20210611_185350_023_fko9ia.jpg",caption= "EDSA-Student", width=150)
	
	# Building out the predication page
	if selection == "Prediction":
		st.write('This is a web app to predict the text category based on\
        the tweet text about climate change. Please enter the\
        text that includes climate change. After that, click on the Classify button at the bottom to\
        see the prediction of the classifier.')

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
			st.success("Status: {}".format(tweet_classifier(prediction)))

def tweet_classifier(prediction):
	if prediction == -1:
		return 'Tweet does not believe in climate change  :-1:'
	elif prediction == 0:  
		return 'Neither the tweet supports nor refuses the believe of climate change  :exclamation:'
	elif prediction == 1:
		return 'Tweet supports the believe of climate change  :stuck_out_tongue_winking_eye:'
	else:
		return 'Tweet links to factual news about climate change  :page_facing_up:'
		

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
