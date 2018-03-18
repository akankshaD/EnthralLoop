import flask
import time
import keras
from keras.models import load_model
import Model

app = flask.Flask(__name__)

@app.route('/')
def index():
	data = flask.request.args
	print(data["post"])
	op = Model.getMovies(data['post'])
	finalop=""
	for i in op:
		finalop+=i+","
	#print(data)
	return flask.jsonify(finalop.rstrip())

if __name__ == '__main__':
	app.run(host = '0.0.0.0')

