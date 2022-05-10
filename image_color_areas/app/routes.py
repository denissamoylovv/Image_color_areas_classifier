from flask import render_template, request,flash,redirect,url_for,jsonify
from image_color_areas.app import app
from image_color_areas.colorize_image import colorize_image_to_base64

@app.route('/', methods=['GET', 'POST'])
def index():
	if request.method == 'POST':
		image_and_variables = {
		    key: float(request.form[key])
		    if key not in ["image_str", "kmean"] else request.form[key]
		    for key in request.form
		}
		image_and_variables['grades']=int(image_and_variables['grades'])

		image_and_variables['image_str']=image_and_variables['image_str'].replace(" ","+")

		flash('Thanks for registering')

		return colorize_image_to_base64(**image_and_variables)
	return render_template('index.html')


