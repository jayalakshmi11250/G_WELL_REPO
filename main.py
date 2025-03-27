from flask import Flask, render_template, request
pip install joblib
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Step 1: Load the saved models and encoders using joblib
rf_model_1 = joblib.load('rf_model_1.pkl')  # Model for Material 1
rf_model_2 = joblib.load('rf_model_2.pkl')  # Model for Material 2
rf_model_3 = joblib.load('rf_model_3.pkl')  # Model for Material 3

crop_name_encoder = joblib.load('crop_name_encoder.pkl')  # Encoder for Crop Name
material_encoder = joblib.load('material_encoder.pkl')  # Encoder for Material Names

# Step 2: Define the function to predict materials based on the crop name
def predict_materials(crop_name):
    """
    Predict the required materials for a given crop name.
    
    Parameters:
    crop_name (str): The name of the crop to predict materials for.
    
    Returns:
    tuple: Materials required for the crop.
    """
    # One-hot encode the crop name
    crop_encoded = crop_name_encoder.transform([[crop_name]])
    
    # Predict the materials for the given crop
    material_1_pred = rf_model_1.predict(crop_encoded)
    material_2_pred = rf_model_2.predict(crop_encoded)
    material_3_pred = rf_model_3.predict(crop_encoded)
    
    # Convert the encoded predictions back to original material names
    material_1 = material_encoder.inverse_transform(material_1_pred)
    material_2 = material_encoder.inverse_transform(material_2_pred)
    material_3 = material_encoder.inverse_transform(material_3_pred)
    
    return material_1[0], material_2[0], material_3[0]

# Step 3: Define routes for the web application
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the crop name from the form input
        crop_name = request.form['crop_name']
        
        # Predict the materials for the given crop name
        try:
            material_1, material_2, material_3 = predict_materials(crop_name)
            return render_template('index.html', crop_name=crop_name, 
                                   material_1=material_1, 
                                   material_2=material_2, 
                                   material_3=material_3)
        except:
            return render_template('index.html', error="Crop name not recognized or invalid input.")
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

