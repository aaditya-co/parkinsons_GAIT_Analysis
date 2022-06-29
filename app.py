import json
import pandas as pd
import numpy as np
import os

from sklearn.metrics import accuracy_score
from video_process_utils import *
from keras.models import load_model
import keras.losses
import keras.metrics
keras.losses.loss = keras.losses.mse
keras.metrics.loss = keras.metrics.mse
from IPython.display import Video
from statsmodels.regression.linear_model import OLSResults
from flask import Flask, jsonify,render_template,request,abort
import tensorflow as tf
import ast
import functools

BROWSERS = ['Mozilla', 'Gecko', 'Chrome', 'Safari']
def no_browsers(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        user_agent = request.headers.get('User-Agent', '')
        if any(agent in user_agent for agent in BROWSERS):
            return abort(400)
        return f(*args, **kwargs)
    return wrapper

config = {
    "DEBUG": True  # run app in debug mode
}




app = Flask(__name__)
app.config.from_mapping(config)

# Test route files ----------------------------------------------
HY_model = tf.keras.models.load_model("Model/HY_Final_Model.h5")
FOG_model = tf.keras.models.load_model("Model/FOG_Final_Model.h5")

#------physionet-----------------------------------------------------

features = ['Time', 'L1' , 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 
            'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 
            'Total_Force_Left', 'Total_Force_Right']
def load_file(filepath):
	dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values
def load_category(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = np.dstack(loaded)
	return loaded
def load_dataset_category(category, prefix=''):
	# load all 19 files as a single array
	filenames = []
	for fname in features:
		filenames.append(category + '/' + fname + '_' + category + '.txt')
  
	# load input data
	X = load_category(filenames, prefix)
	# load class output
	y = load_file(prefix + 'y_'+ category +'.txt')
	hy = load_file(prefix + 'hyscore_'+ category +'.txt')
	return X, y, hy


def load_dataset_physionet(prefix=''):
	# load all train
	trainX, trainy, trainHY = load_dataset_category('train', prefix)
	print(trainX.shape, trainy.shape, trainHY.shape)
	# load all test
	testX, testy, testHY = load_dataset_category('test', prefix)
	print(testX.shape, testy.shape, testHY.shape)
 
	# zero-offset class values
	trainy = trainy - 1
	testy = testy - 1
	# one hot encode y
	trainy = tf.keras.utils.to_categorical(trainy)
	trainHY = tf.keras.utils.to_categorical(trainHY)
	testy = tf.keras.utils.to_categorical(testy)
	testHY = tf.keras.utils.to_categorical(testHY)  
	return trainX, trainy, trainHY, testX, testy, testHY
# trainX, trainy, trainHY, testX, testy, testHY = load_dataset_physionet('Final/')

#------daphnet----------------------------------------------------- 
class_var = 'annontations'
features_daphnet = ['Time', 'ankle-x', 'ankle-y', 'ankle-z', 
            'thigh-x', 'thigh-y', 'thigh-z',
            'trunk-x', 'trunk-y', 'trunk-z', class_var]
def load_file_daphnet(filepath):
	dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values
def load_category_daphnet(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file_daphnet(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = np.dstack(loaded)
	return loaded
def load_dataset_category_daphnet(category, prefix=''):
	# load all 10 files as a single array
  filenames = []
  for fname in features_daphnet:
    if fname != 'Time' and fname != class_var:
      filenames.append(category + '/' + fname + '_' + category + '.txt')
  
  # load input data
  X = load_category_daphnet(filenames, prefix)
  # load class output
  y = load_file_daphnet(prefix + 'y_'+ category +'.txt')
  return X, y
def load_dataset_daphnet(prefix=''):
  # load all train
  trainX, trainy = load_dataset_category_daphnet('train', prefix)
  print(trainX.shape, trainy.shape)
  # load all test
  testX, testy = load_dataset_category_daphnet('test', prefix)
  print(testX.shape, testy.shape)

  # zero-offset class values
  trainy = trainy - 1
  testy = testy - 1

  neg, pos = np.bincount(trainy.reshape(1, trainy.shape[0])[0])
  total = neg + pos
  print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(total, pos, 100 * pos / total))
  # one hot encode y
  trainy = tf.keras.utils.to_categorical(trainy)
  testy = tf.keras.utils.to_categorical(testy)

  # print(trainX.shape, trainy.shape, testX.shape, testy.shape)
  return trainX, trainy, testX, testy, neg, pos, total

def final_prediction(HY_model ,FOG_model, pd_data, fog_data):
  # pd_data - 100 x 19
  # fog_data - 128 x 9
  hy_pred = np.argmax(HY_model.predict(np.expand_dims(pd_data, axis=0)))
  fog_pred = np.argmax(FOG_model.predict(np.expand_dims(fog_data, axis=0)))

  
  hy_pred = str(hy_pred)
  fog_pred = str(fog_pred)

  return dict(HY=hy_pred, FOG=fog_pred)

#-----------------------------------------------------------------


@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def data():
    # path_example  = out/07337701-processed/
    if request.method == 'POST':
        path = request.form['data']
        frames = convert_json2csv(path)
        pd.DataFrame(frames)
        processed_videos = []
        processed_video_segments = []
        centered_filtered = process_video_and_add_cols(frames)
        centered_filtered_noswap = process_video_and_add_cols(frames,
                                        swap_orientation=False)

        res = (get_prediction(centered_filtered,"GDI"))
    return render_template('symptoms.html',res=res)



@app.route('/symptoms') 
@no_browsers
def symptoms():
    return render_template('symptoms.html')

@app.route('/test')
def test_test():
  return render_template('test.html')

@app.route('/test', methods=['POST'])
def test_post():
    if request.method == 'POST':
      trainX, trainy, trainHY, testX, testy, testHY = load_dataset_physionet('Final/')
      fog_trainX, fog_trainy, fog_testX, fog_testy, neg, pos, total = load_dataset_daphnet('Final_D/')
      pd_data_str = request.form['pd_data']
      pd_data = ast.literal_eval(pd_data_str)
      fog_data_str = request.form['fog_data']
      fog_data = ast.literal_eval(fog_data_str)
      final_prediction(HY_model,FOG_model, pd_data,fog_data)
    return render_template('result.html',jsonfile=json.dumps(final_prediction(HY_model,FOG_model, pd_data,fog_data)))

@app.route('/result')
@no_browsers
def result_get():
    return render_template('result.html')

# OPenpose Functions------------------------------------------------------------------------------------
def convert_json2csv(json_dir):

    resL = np.zeros((2000,75))
    resL[:] = np.nan
    for frame in range(1,2000):
        # test_image_json = '%sinput_%s_keypoints.json' %\
        #     (json_dir, str(frame).zfill(12)) #input_000000000000_keypoints.json
        temp_dir = extract(json_dir)


        test_image_json = json_dir+temp_dir+'-processed_%s_keypoints.json' %\
            (str(frame).zfill(12)) #input_000000000000_keypoints.json

        if not os.path.isfile(test_image_json):
            break
        with open(test_image_json) as data_file:  
            data = json.load(data_file)

        for person in data['people']:
            keypoints = person['pose_keypoints_2d']
            xcoords = [keypoints[i] for i in range(len(keypoints)) if i % 3 == 0]
            counter = 0
            resL[frame-1,:] = keypoints
            break
    check = np.apply_along_axis(lambda x: np.any(~np.isnan(x)),1,resL)
    for i in range(len(check)-1,-1,-1):
        if check[i]:
            break
    return resL[:i+1]

def get_prediction(centered_filtered, col, side = None):
    model = load_model("models/{}_best.pb".format(col))
    correction_model = OLSResults.load("models/{}_correction.pb".format(col))
    maps = {
        "GDI": (25, 100)
    }

    def undo_scaling(y,target_min,target_range):
        return y*target_range+target_min

    preds = []

    video_len = centered_filtered.shape[0]
    
    cols = x_columns
    if side == "L":
        cols = x_columns_left
    if side == "R":
        cols = x_columns_right

    samples = []
    for nstart in range(0,video_len-124,31):
        samples.append(centered_filtered[nstart:(nstart+124),cols])
    X = np.stack(samples)
    
    p = model.predict(X)[:,0]
    p = undo_scaling(p, maps[col][0], maps[col][1])
    p = np.transpose(np.vstack([p,np.ones(p.shape[0])]))
    p = correction_model.predict(pd.DataFrame(p))

    # predict PD on the basis of GDI values
    gdi = np.mean(p)
    # normal range of gait deviation index is 80+ 
    if gdi > 80:
        return 0
    else:
        return 1

def extract(path):
    num_p = ""
    for c in path:
        if c.isdigit():
            num_p = num_p + c
    
    return num_p

if __name__ == '__main__':
   app.run()
