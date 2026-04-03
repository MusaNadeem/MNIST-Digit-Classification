import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
import tensorflow as tf

app = Flask(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'mnist_model.h5')
# Need to use keras load_model as requested:
model = tf.keras.models.load_model(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Determine if base64 contains header, if so strip it
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        decoded_image = base64.b64decode(image_data)
        
        # Open with PIL, convert to grayscale
        img = Image.open(io.BytesIO(decoded_image)).convert('L')
        
        # Resize to 28x28
        img = img.resize((28, 28))
        
        # Convert to numpy array and normalize to 0-1
        img_array = np.array(img).astype('float32') / 255.0
        
        # Reshape to (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)
        
        # Run prediction
        predictions = model.predict(img_array)
        probs = predictions[0].tolist()
        digit = int(np.argmax(probs))
        confidence = float(np.max(probs)) * 100
        
        return jsonify({
            'digit': digit,
            'confidence': confidence,
            'all_probs': probs
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
