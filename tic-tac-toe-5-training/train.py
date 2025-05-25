import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pickle
import numpy as np
from numba import prange, njit
import importlib
import sys
import random
import threading
from multiprocessing import Process
import time
import tensorflow as tf

import ModelConfig as mc
# importlib.reload(mc)
import TensorflowModel as tm
# importlib.reload(tm)
import ModelStrategies as ms
# importlib.reload(ms)


def train(model):
	x_train, y_train = None, None
	x_test, y_test = None, None
	strategy = model.config.input_strategy_number
	with open(f'datasets/train_{strategy:02d}.pkl', 'rb') as file:
		x_train, y_train = pickle.load(file)
	with open(f'datasets/test_{strategy:02d}.pkl', 'rb') as file:
		x_test, y_test = pickle.load(file)

	print('Starting', model.config.model_name)
	# start = time.time()
	# for i in range(3):
	while True:
		model.model.fit(
			x_train, y_train, epochs=50, 
			verbose=0, 
			shuffle=True,
			validation_data=(x_test, y_test), 
			validation_split=0.1)
		model.save_model()
		result = model.model.evaluate(x_test, y_test, verbose=0)
		with open(f'{model.config.model_name}/train.log', 'a') as logs:
			logs.write(str(result) + '\n')
	# end = time.time()
	print(
		'Finishing', 
	    model.config.model_name, 
		f'Training took {end-start:.1f} seconds', 
		sep='\n>>> ')
	# startNext()
	
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
	print(gpu)
	tf.config.experimental.set_memory_growth(gpu, True)
	
if __name__ == '__main__':
	models = []
	processes = []
	for i in range(1, 11):
		model_name = f'Size_81_81_27/Inputs_{i:02d}'
		config = mc.ModelConfig(
			hidden_structure=[81, 81, 27], 
			input_strategy_number=i, 
			output_strategy_number=1, 
			use_bias=True,
			use_attention=False,
			model_name=model_name
		)
		t_model = tm.TensorflowModel(config)
		if (os.path.isdir(model_name)):
			t_model.load_model()
		else:
			t_model.save_model()
		t_model.model.summary()
		models.append(t_model)
	for t_model in models:
		# train(t_model)
		t = Process(
			target=train, 
			args=(t_model,))
		processes.append(t)

	start = time.time()
	for i in range(len(processes)):
		processes[i].start()
	
	for i in range(len(processes)):
		processes[i].join()
	end = time.time()
	print(f"All trainings took {end-start:.1f} seconds")
	