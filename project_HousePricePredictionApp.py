import streamlit as st
import joblib

# Load the trained machine learning model using joblib
model = joblib.load('Price_Prediction_model_compressed.pkl')

# Set page config for a better title and layout
st.set_page_config(page_title="USA House Price Prediction App", layout="wide")

# Title and a captivating introduction
st.title('🏡 USA House Price Prediction App')
st.markdown("""
Welcome to the **USA House Price Prediction App**! 🎉  
This app uses advanced machine learning models to predict house prices based on features like area, number of rooms, amenities, and more. 
Want to know how much a property costs? Just fill in the details below, and let the app predict the price for you! 🔮
""")

# Adding an image for a more appealing introduction
st.image("https://github.com/MagHammond91/python_project-HousePricePrediction-/blob/e71fbc65706cb63214f1fadb8abcd71a6555598f/images/housing.image.jpeg", caption="A beautiful home awaits you!")

# Form for user inputs
with st.form('Myform'):
    col1, col2, col3 = st.columns(3)
    
    # Input fields for the user
    area = col1.number_input('Area size (in square feet)', min_value=0, step=5)
    bedrooms = col1.selectbox('Select number of Bedrooms', [1, 2, 3, 4, 5, 6])
    bathrooms = col1.selectbox('Select number of Bathrooms', [1, 2, 3, 4])
    stories = col1.selectbox('Select number of Stories', [1, 2, 3, 4])

    # Additional feature options with yes/no answers
    def map_input(input_value, true_value): return 1 if input_value.lower() == true_value else 0
    
    main_road_input = col2.selectbox('Do you want a property located on the Main road?', ['Yes', 'No'])
    main_road_numeric = map_input(main_road_input, 'yes')

    guestroom_input = col2.selectbox('Do you want a property with a Guestroom?', ['Yes', 'No'])
    guestroom_numeric = map_input(guestroom_input, 'yes')

    basement_input = col2.selectbox('Do you want a property with a Basement?', ['Yes', 'No'])
    basement_numeric = map_input(basement_input, 'yes')

    hotwater_heating_input = col2.selectbox('Do you want a property with hot water heating?', ['Yes', 'No'])
    hotwater_heating_numeric = map_input(hotwater_heating_input, 'yes')

    air_conditioning_input = col3.selectbox('Do you want a property with Air Conditioning?', ['Yes', 'No'])
    air_conditioning_numeric = map_input(air_conditioning_input, 'yes')

    parking = col3.selectbox('Select number of Parking Lots', [0, 1, 2, 3])

    pref_area_input = col3.selectbox('Do you want a property in a preferred area?', ['Yes', 'No'])
    pref_area_numeric = map_input(pref_area_input, 'yes')

    def map_furnishing_status(input_value):
        if input_value.lower() == 'furnished':
            return 2
        elif input_value.lower() == 'semi-furnished':
            return 1
        return 0

    furnishing_status_input = col3.selectbox('Select furnishing status:', ['Furnished', 'Semi-Furnished', 'Unfurnished'])
    furnishing_status_numeric = map_furnishing_status(furnishing_status_input)

    submit = st.form_submit_button('Predict Price')

# Process the inputs and display the result when the form is submitted
if submit:
    input_data = [[area, bedrooms, bathrooms, stories, main_road_numeric, guestroom_numeric,
                   basement_numeric, hotwater_heating_numeric, air_conditioning_numeric,
                   parking, pref_area_numeric, furnishing_status_numeric]]

    predicted_price = model.predict(input_data)
    n = '${:,.2f}'.format(predicted_price[0])
    
    # Display the prediction
    st.success(f'🏠 **Predicted House Price**: {n}')

    # Add some engaging content after prediction
    st.markdown("""
    ### That's the estimated price based on your input.  
    Ready to find your dream home? 🏡  
    Explore various properties and start your journey to a new home today!  
    """)
    
    st.image("https://your_image_url.com/another_image.jpg", caption="Your new home could be just around the corner!")






#To run the web app interface, open and type this in the terminal 'streamlit run project_HousePricePredicionApp.py'
