from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO

app = Flask(_name_)

model = load_model('car_bike_classifier.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_file = request.files['image'].read()
        
        img = image.load_img(BytesIO(image_file), target_size=(64, 64))
        
        img_array = image.img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array_expanded)
        
        result = {
            'class': 'car' if prediction[0][0] > 0.5 else 'bike',
            'confidence': float(prediction[0][0])
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if _name_ == '_main_':
    app.run(host='0.0.0.0', port=3300)