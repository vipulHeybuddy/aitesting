from flask import Flask, render_template, jsonify, request
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from roboflow import Roboflow
from matplotlib.path import Path
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import cloudinary
import cloudinary.uploader



app = Flask(__name__)
# After creating the Flask app, you can make all APIs allow cross-origin access.
CORS(app)
# or a specific API



# Initialize Roboflow
rf = Roboflow(api_key="zpylwepWJURMjyaus8J7")
project = rf.workspace().project("room-lyc7i")
model = project.version(1).model

api_key = '783677746271476'
api_secret = "2SX3P76y3PwqMPHhkrIWEm_msbs"
cloud_name = "dyx9ocdno"

def create_mask_from_points(image_size, points):
    y, x = np.mgrid[:image_size[1], :image_size[0]]
    points = np.array(points)
    path = Path(points)
    mask = path.contains_points(np.vstack((x.ravel(), y.ravel())).T)
    mask = mask.reshape((image_size[1], image_size[0]))
    return mask

def process_image(room_image_path, texture_image_path):
    print("hello" , room_image_path)
    room_image = Image.open(room_image_path)
    texture_image = Image.open(texture_image_path)

    # JSON data
    json_data = model.predict(room_image_path).json()

    # Extract floor coordinates
    floor_data = json_data['predictions'][0]
    floor_points = floor_data['points']

    # Create a list of tuples for the points
    floor_coordinates = [(point['x'], point['y']) for point in floor_points]

    # Create mask for the floor
    mask = create_mask_from_points(room_image.size, floor_coordinates)

    # Get the dimensions of the room image
    room_width, room_height = room_image.size

    #continuous image
    resized_texture = texture_image.resize(room_image.size)
    
    # Repeat the texture to cover the entire room
    # repeated_texture = np.tile(np.array(texture_image),
    # (room_height // texture_image.size[1] + 1, room_width // texture_image.size[0] + 1, 1))
    # repeated_texture = repeated_texture[:room_height, :room_width, :]
    # # Use the mask to combine the images


    room_with_texture = np.array(room_image)
    room_with_texture[mask] = np.array(resized_texture)[mask]
    
    #Repeated texture code
    # room_with_texture[mask] = repeated_texture[mask]

    # Convert back to an image
    room_with_texture_image = Image.fromarray(room_with_texture)

    # Save the result
    output_image_path = 'static/newfinal.jpg'  # Save in the static folder to serve via Flask
    room_with_texture_image.save(output_image_path)

    return output_image_path

@app.route('/get', methods=['GET', 'POST'])
@cross_origin()
def index():
    if request.method == 'POST':
        texture_name = request.form['texture_name']
        print(texture_name)

        i = int(texture_name)
        Image_arr = ['t1.jpg', 't2.jpg', 't3.jpg', 't4.jpg']
        texture_image_path = 'uploads/' + Image_arr[i]
        
        # if request.files['room_image'].filename == "":
        room_image_path = 'uploads/' + secure_filename(request.files['room_image'].filename)

       

        # Use the locally stored image in the process_image function
        output_image_path = process_image(room_image_path, texture_image_path)

        return render_template('index.html', result_image=output_image_path)

    return render_template('index.html')



@app.route("/upload", methods=['POST'])
def upload_file():
    app.logger.info('in upload route')

    cloudinary.config(cloud_name=cloud_name, api_key=api_key, api_secret=api_secret)
    upload_result = None
    print("Request Body:")
    print(request.get_data(as_text=True))
    
    if request.method == 'POST':
        file_to_upload = request.files['file']
        app.logger.info('%s file_to_upload', file_to_upload)

        if file_to_upload:
            upload_result = cloudinary.uploader.upload(file_to_upload)  # Use cloudinary.uploader.upload directly
            app.logger.info(upload_result)
            print(upload_result)
            return jsonify(upload_result)
    
    return jsonify({'error': 'No file provided'})

if __name__ == "__main__":
    app.run(debug=True)