# import from numpy.core.fromnumeric import prod
# import tensorflow as tf
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import streamlit as st

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Import the Dataset 
skincare = pd.read_csv("export_skincare.csv", encoding='utf-8', index_col=None)

# Header
st.set_page_config(page_title="Skin Care Product Recommendation Application", page_icon=":blossom:", layout="wide",)

# Displaying the main page
st.title("Skin Care Product Recommendation Application :sparkles")

st.write('---') 

# #displaying a local video file

# video_file = open("skincare.mp4", "rb").read()
# st.video(video_file, start_time = 1) #displaying the video 


st.write('---') 

st.write(
    """
    **"The Skin Care Product Recommendation Application is an implementation of a Machine Learning project that provides skin care product recommendations based on your skin type and concerns. You can input your skin type, concerns, and desired benefits to get the most suitable skin care product recommendations.**
    """)  
st.write('---') 

first,last = st.columns(2)

# Choose a product product type category
# pt = product type
category = first.selectbox(label='Product Category : ', options= skincare['product_type'].unique() )
category_pt = skincare[skincare['product_type'] == category]

# Choose a skintype
# st = skin type
skin_type = last.selectbox(label='Your Skin Type : ', options= ['Normal', 'Dry', 'Oily', 'Combination', 'Sensitive'] )
category_st_pt = category_pt[category_pt[skin_type] == 1]

# pilih keluhan
prob = st.multiselect(label='Your Skin Concerns : ', options= ['Dull skin', 'Acne', 'Acne Scars','Large Pores', 'Dark Spots', 'Fine Lines and Wrinkles', 'Blackheads', 'Uneven Skin Tone', 'Redness', 'Sagging Skin'] )

# Choose notable_effects
# From the products already filtered based on product type and skin type (category_st_pt), we will extract unique values from the notable_effects column."
opsi_ne = category_st_pt['notable_effects'].unique().tolist()
# Unique notable_effects values are stored in the variable opsi_ne and used as options in the multiselect wrapped in the variable selected_options below
selected_options = st.multiselect('Desired Benefits: ',opsi_ne)
# The result of selected_options is stored in the variable
category_ne_st_pt = category_st_pt[category_st_pt["notable_effects"].isin(selected_options)]

# Choose product
#Products that have already been filtered and are in the variable filtered_df are then further filtered to extract unique values based on product_name and stored in the variable opsi_pn 
opsi_pn = category_ne_st_pt['product_name'].unique().tolist()
# buat sebuah selectbox yang berisi pilihan produk yg sudah di filter di atas
product = st.selectbox(label='Products Recommended for You', options = sorted(opsi_pn))
# The Product variable above will store a product that will be used to display recommendations for other similar products.
## MODELLING with Content Based Filtering
# Inisialisasi TfidfVectorizer
tf = TfidfVectorizer()

# Performing IDF calculation on the 'notable_effects' data
tf.fit(skincare['notable_effects']) 

# Mapping an array from integer feature indices to feature names
tf.get_feature_names()

# Performing fit and then transforming into a matrix form"
tfidf_matrix = tf.fit_transform(skincare['notable_effects']) 

# Viewing the size of the tf-idf matrix
shape = tfidf_matrix.shape

# Converting the tf-idf vector into a matrix using the todense() function
tfidf_matrix.todense()

# Creating a dataframe to view the tf-idf matrix
# Columns are filled with the desired effects
# Rows are filled with product names
pd.DataFrame(
    tfidf_matrix.todense(), 
    columns=tf.get_feature_names(),
    index=skincare.product_name
).sample(shape[1], axis=1).sample(10, axis=0)

# Calculating cosine similarity on the tf-idf matrix
cosine_sim = cosine_similarity(tfidf_matrix) 

# Creating a dataframe from the cosine_sim variable with rows and columns as product names
cosine_sim_df = pd.DataFrame(cosine_sim, index=skincare['product_name'], columns=skincare['product_name'])

# Viewing the similarity matrix for each product name
cosine_sim_df.sample(7, axis=1).sample(10, axis=0)

# Creating a function to get recommendations
def skincare_recommendations(nama_produk, similarity_data=cosine_sim_df, items=skincare[['product_name', 'produk-href','price', 'description']], k=5):
    
    # Retrieving data using argpartition to perform indirect partitioning along the given axis
    #The dataframe is converted to numpy
    # Range(start, stop, step)
    index = similarity_data.loc[:,nama_produk].to_numpy().argpartition(range(-1, -k, -1))
    
    # Retrieving data with the highest similarity from the available index
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    
    # Drop nama_produk so that the searched product name does not appear in the recommendation list.
    closest = closest.drop(nama_produk, errors='ignore')
    df = pd.DataFrame(closest).merge(items).head(k)
    return df

# Membuat button untuk menampilkan rekomendasi
model_run = st.button('Find Other Product Recommendations!')
# Getting recommendations
if model_run:
    st.write('Here are other similar product recommendations based on your preferences')
    st.write(skincare_recommendations(product))
    st.snow()
