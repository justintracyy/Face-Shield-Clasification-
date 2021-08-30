from flask import Flask, render_template, request, url_for
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


app = Flask(__name__)

dic = {0 : 'No Facehield', 1 : 'With Face Shield'}

model = load_model('newmodelfs.h5')

model.make_predict_function()

def predict_label(img_path):
	i = image.load_img(img_path, target_size=(200,200))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 200,200,3)
	px = model.predict(i)
	p =np.argmax(px,axis=1)
	return dic[p[0]]



# routes
@app.route("/", methods=['GET'])
def main():
	return render_template("home1.html")

@app.route("/classifier",  methods=['GET'])
def classifier():
  return render_template('classifier1.html')

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("classifier1.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)