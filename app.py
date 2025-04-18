import streamlit as st
from streamlit_option_menu import option_menu
from numpy.core.fromnumeric import prod
import tensorflow as tf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# Import the Dataset 
skincare = pd.read_csv("export_skincare.csv", encoding='utf-8', index_col=None)

# Header
st.set_page_config(page_title="Skin Care Recommender System", page_icon=":rose:", layout="wide",)

# 1=sidebar menu, 2=horizontal menu, 3=horizontal menu w/ custom menu
EXAMPLE_NO = 2

def streamlit_menu(example=1):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",  # required
                options=["Skin Care", "Get Recommendation", "About Skin Care"],  # required
                icons=["house", "stars", "book"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
            )
        return selected

    if example == 2:
        # 2. horizontal menu w/o custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Skin Care", "Get Recommendation", "About Skin Care"],  # required
            icons=["house", "stars", "book"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
        return selected

    if example == 3:
        # 2. horizontal menu with custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Skin Care", "Get Recommendation", "About Skin Care"],  # required
            icons=["house", "stars", "book"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "green"},
            },
        )
        return selected


selected = streamlit_menu(example=EXAMPLE_NO)

if selected == "Skin Care":
    st.title(f"{selected} Product Recommender :sparkles:")
    st.write('---') 

    st.write(
        """
        **The Skin Care Product Recommendation Application is one implementation of Machine Learning that can provide skin care product recommendations based on your skin type and concerns.**
        """)
    
    #displaying a local video file

    # video_file = open("skincare.mp4", "rb").read()
    # st.video(video_file, start_time = 1) #displaying the video 
    
    # st.write(' ') 
    # st.write(' ')
    st.write(
        """You will receive skincare product recommendations from various cosmetic brands, with a total of more than 1,200 products tailored to your skin needs.
        There are five skincare product categories and five different skin types, along with concerns and benefits that users may want from their products.
        This recommendation application is simply a system that provides suggestions based on the data you input—it's not a scientific consultation.
        Please select the 'Get Recommendation' page to start receiving recommendations, or choose 'About Skin Care' to explore skincare tips and tricks.
        """)
    
    st.write(
        """
        **Happy trying! 😊**
        """)
    
    
    st.info('Credit: Created by Team Matrix')

if selected == "Get Recommendation":
    st.title(f"Let's {selected}")
    
    st.write(
        """
        **To receive recommendations, please enter your skin type, concerns, and desired benefits to get the right skincare product recommendations**
        """) 
    
    st.write('---') 

    first,last = st.columns(2)

    # Choose a product product type category
    # pt = product type
    category = first.selectbox(label='Product Category : ', options= skincare['product_type'].unique() )
    category_pt = skincare[skincare['product_type'] == category]

    # Choose a skin type
    # st = skin type
    skin_type = last.selectbox(label='Your Skin Type : ', options= ['Normal', 'Dry', 'Oily', 'Combination', 'Sensitive'] )
    category_st_pt = category_pt[category_pt[skin_type] == 1]

    # choose concerns
    prob = st.multiselect(label='Skin Problems : ', options= ['Dull Skin', 'Acne', 'Acne Scars','Large Pores', 'Dark Spots', 'Fine Lines and Wrinkles', 'Blackheads', 'Uneven Skin Tone', 'Redness', 'Sagging Skin'] )

    # Choose notable_effects
    #From the products already filtered based on product type and skin type (category_st_pt), we will extract unique values from the 'notable_effects' column.
    opsi_ne = category_st_pt['notable_effects'].unique().tolist()
    # Unique notable_effects values are stored in the variable opsi_ne and used as options in the multiselect wrapped in the variable selected_options below.
    selected_options = st.multiselect('Notable Effects : ',opsi_ne)
    # The result of selected_options is stored in the variable category_ne_st_pt.
    category_ne_st_pt = category_st_pt[category_st_pt["notable_effects"].isin(selected_options)]

    # Choose product
    # Products that have already been filtered and are in the variable filtered_df are then further filtered to extract unique values based on product_name and stored in the variable opsi_pn.
    opsi_pn = category_ne_st_pt['product_name'].unique().tolist()
    # Create a selectbox containing the product options that have already been filtered above
    product = st.selectbox(label='Products Recommended for You', options = sorted(opsi_pn))
    # The product variable above will store a product that will be used to display recommendations for other similar products

    ## MODELLING with Content Based Filtering
    # Inisialisasi TfidfVectorizer
    tf = TfidfVectorizer()

    # Performing IDF calculation on the 'notable_effects' data
    tf.fit(skincare['notable_effects']) 

    # Mapping an array from integer feature indices to feature names
    tf.get_feature_names_out()

    # Performing fit and then transforming into a matrix form
    tfidf_matrix = tf.fit_transform(skincare['notable_effects']) 

    # Viewing the size of the tf-idf matrix
    shape = tfidf_matrix.shape

    # Converting the tf-idf vector into a matrix using the todense() function
    tfidf_matrix.todense()

#     Creating a dataframe to view the tf-idf matrix
# Columns are filled with the desired effects
# Rows are filled with product names
    pd.DataFrame(
        tfidf_matrix.todense(), 
        columns=tf.get_feature_names_out(),
        index=skincare.product_name
    ).sample(shape[1], axis=1).sample(10, axis=0)

    # Calculating cosine similarity on the tf-idf matrix
    cosine_sim = cosine_similarity(tfidf_matrix) 

    # Creating a dataframe from the cosine_sim variable with rows and columns as product names
    cosine_sim_df = pd.DataFrame(cosine_sim, index=skincare['product_name'], columns=skincare['product_name'])

    # Viewing the similarity matrix for each product name
    cosine_sim_df.sample(7, axis=1).sample(10, axis=0)

    # Creating a function to get recommendations
    def skincare_recommendations(nama_produk, similarity_data=cosine_sim_df, items=skincare[['product_name', 'brand', 'description']], k=5):

        # Retrieving data using argpartition to perform indirect partitioning along the given axis    
        # the dataframe is converted to numpy
        # Range(start, stop, step)
        index = similarity_data.loc[:,nama_produk].to_numpy().argpartition(range(-1, -k, -1))

        # Retrieving data with the highest similarity from the available index
        closest = similarity_data.columns[index[-1:-(k+2):-1]]

        # Drop nama_produk so that the searched product name does not appear in the recommendation list
        closest = closest.drop(nama_produk, errors='ignore')
        df = pd.DataFrame(closest).merge(items).head(k)
        return df

    # Creating a button to display recommendations
    model_run = st.button('Find Other Product Recommendations!')
    # Mendapatkan rekomendasi
    if model_run:
        st.write('Here are other similar product recommendations based on your preferences')
        st.write(skincare_recommendations(product))
    
    
if selected == "About Skin Care":
    st.title(f"Take a Look at {selected}")
    st.write('---') 

    st.write(
        """
        **Here are tips and tricks you can follow to maximize the use of skincare products.**
        """) 
    
    image = Image.open('imagepic.jpg')
    st.image(image, caption='About Skin Care')
    

    
    st.write(
        """
        **1. Facial Wash**
        """)
    st.write(
        """
        **- Use the facial wash product that has been recommended or that suits you.**
        """)
    st.write(
        """
        **- Wash your face a maximum of twice a day, in the morning and at night before bed. Washing your face too often can strip the skin's natural oils. For those with dry skin, it's okay to use just water in the morning.**
        """)
    st.write(
        """
        **- Do not rub your face harshly as it can remove the skin's natural protective barrier**
        """)
    st.write(
        """
        **- The best way to clean your skin is by using your fingertips for 30-60 seconds with circular and massaging motions.**
        """)
    
    st.write(
        """
        **2. Toner**
        """)
    st.write(
        """
        **- Use the toner that has been recommended or that suits you.**
        """)
    st.write(
        """
        **- Pour the toner onto a cotton pad and gently wipe it on your face. For better results, use two layers of toner: the first with a cotton pad and the second with your hands to help it absorb better.**
        """)
    st.write(
        """
        **- Use toner after washing your face.**
        """)
    st.write(
        """
        **- For those with sensitive skin, try to avoid skincare products that contain fragrance.**
        """)
    
    st.write(
        """
         **3. Serum**
        """)
    st.write(
        """
        **- Use the serum that has been recommended or that suits you for better results.**
        """)
    st.write(
        """
        **- Serum should be applied after your face is completely clean to ensure the serum's ingredients are fully absorbed.**
        """)
    st.write(
        """
        **- Use serum in the morning and at night before bed.**
        """)
    st.write(
        """
        **- Choose a serum based on your needs, such as reducing acne scars, dark spots, anti-aging, or other benefits.**
        """)
    st.write(
        """
        **- To apply serum for better absorption, pour it into your palm, gently pat it onto your face, and wait until it is fully absorbed.**
        """)
    
    st.write(
        """
        **4. Moisturizer**
        """)
    st.write(
        """
        **- Use the moisturizer that has been recommended or that suits you for better results.**
        """)
    st.write(
        """
        **- Moisturizer is an essential skincare product because it locks in moisture and nutrients from the serum you have applied.**
        """)
    st.write(
        """
        **- For better results, use different moisturizers in the morning and at night. Morning moisturizers usually contain sunscreen and vitamins to protect the skin from UV rays and pollution, while night moisturizers contain active ingredients to help skin regeneration during sleep.**
        """)
    st.write(
        """
        **- Allow a gap of 2-3 minutes between applying serum and moisturizer to ensure the serum is fully absorbed into the skin.**
        """)
    
    st.write(
        """
        **5. Sunscreen**
        """)
    st.write(
        """
        **- Use the sunscreen that has been recommended or that suits you for better results.**
        """)
    st.write(
        """
        **- Sunscreen is the key to all skincare products because it protects the skin from harmful UVA and UVB rays, as well as blue light. All skincare products will not be effective without protection for the skin.**
        """)
    st.write(
        """
        **- Use sunscreen approximately the length of your index and middle fingers to maximize protection.**
        """)
    st.write(
        """
        **- Reapply sunscreen every 2-3 hours or as needed.**
        """)
    st.write(
        """
        **- Continue using sunscreen even indoors because sunlight after 10 AM can penetrate windows, even on cloudy days.**
        """)
    
    st.write(
        """
        **6. Do not frequently change skincare products**
        """)
    st.write(
        """
        **Frequently changing skincare products can stress your skin as it has to adapt to new ingredients. As a result, the benefits may not be fully realized. Instead, use skincare products consistently for months to see results.**
        """)
    
    st.write(
        """
        **7. Consistency**
        """)
    st.write(
        """
        **The key to skincare is consistency. Be diligent and persistent in using skincare products because the results are not instant.**
        """)
    st.write(
        """
        **8. Your face is an asset**
        """)
    st.write(
        """
        **The diversity of human appearances is a gift from the Creator. Take good care of this gift as a form of gratitude. Choose products and skincare routines that suit your skin's needs. Using skincare products early is like investing in your future.**
        """)