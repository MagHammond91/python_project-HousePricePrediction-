import streamlit as st
import pickle

# Load the trained machine learning model
pickle_in = open('Price_Prediction_model.h5','rb')
model = pickle.load(pickle_in)

st.title('USA House Price Prediction App')


with st.form('Myform'):
        col1, col2, col3 = st.columns(3)
        area = col1.number_input('Area size(in square feet)',min_value = 0, step=5)
        bedrooms = col1.selectbox('Select no. of Bedrooms', [1,2,3,4,5,6])
        bathrooms = col1.selectbox('Select no. of Bathrooms', [1,2,3,4])
        stories = col1.selectbox('Select no. of Stories', [1,2,3,4])

        def map_main_road(main_road_input):return 1 if main_road_input.lower() == 'yes' else 0
        main_road_input = col2.selectbox('Do you want a property located on the Main road?', ['Yes', 'No'])
        main_road_numeric = map_main_road(main_road_input)

        def map_guestroom(guestroom_input):return 1 if guestroom_input.lower() == 'yes' else 0
        guestroom_input = col2.selectbox('Do you want a property with a Guestroom?', ['Yes', 'No'])
        guestroom_numeric = map_guestroom(guestroom_input)

        def map_basement(basement_input):return 1 if basement_input.lower() == 'yes' else 0
        basement_input = col2.selectbox('Do you want a property with a Basement?', ['Yes', 'No'])
        basement_numeric = map_basement(basement_input)

        def map_hotwater_heating(hotwater_heating_input):return 1 if hotwater_heating_input.lower() == 'yes' else 0
        hotwater_heating_input = col2.selectbox('Do you want a property with a hot water heating?', ['Yes', 'No'])
        hotwater_heating_numeric = map_hotwater_heating(hotwater_heating_input)

        def map_air_conditioning(air_conditioning_input):return 1 if air_conditioning_input.lower() == 'yes' else 0
        air_conditioning_input = col3.selectbox('Do you want a property with an Air conditioning?', ['Yes', 'No'])
        air_conditioning_numeric = map_air_conditioning(air_conditioning_input)

        parking = col3.selectbox('Select no. of Parking lot', [0,1,2,3])

        def map_pref_area(pref_area_input):return 1 if pref_area_input.lower() == 'yes' else 0
        pref_area_input = col3.selectbox('Do you want a property within a preferred area?', ['Yes', 'No'])
        pref_area_numeric = map_pref_area(pref_area_input)

        def map_furnishing_status(furnishing_status_input):
            if furnishing_status_input.lower() == 'furnished':
                return 2
            elif furnishing_status_input.lower() == 'semi-furnished':
                return 1
            else:
                return 0
        furnishing_status_input = col3.selectbox('Select furnishing status:', ['Furnished', 'Semi-Furnished', 'Unfurnished'])
        furnishing_status_numeric = map_furnishing_status(furnishing_status_input)

        submit = st.form_submit_button('Process')

# Process the inputs and provide feedback
if submit:
    # Here you can add the code to process the inputs and generate predictions
    # For now, let's just display the inputs as feedback
    input_data = [[area, bedrooms, bathrooms, stories, main_road_numeric, guestroom_numeric,
                   basement_numeric, hotwater_heating_numeric, air_conditioning_numeric,
                   parking, pref_area_numeric, furnishing_status_numeric]]
    
    predicted_price = model.predict(input_data)
    n = '${}'.format(predicted_price[0])
    st.success(f'Predicted Price: {n}')





#To run the web app interface, enter this in the terminal 'streamlit run HousePricePredicionApp.py'
