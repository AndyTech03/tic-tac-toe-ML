import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import shutil
import datetime
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
import GameBoard as gb
# importlib.reload(gb)

def print_board(x_moves, o_moves, player_is_x):
	board = [
		'  ', '  ', '  ',
		'  ', '  ', '  ',
		'  ', '  ', '  ',
	]
	for i in x_moves:
		board[i] = '><'
	for i in o_moves:
		board[i] = '()'
	if player_is_x and len(o_moves) > 0:
		board[o_moves[-1]] = '[]'
	elif player_is_x == False and len(x_moves) > 0:
		board[x_moves[-1]] = '}{'
	print(*board[:3], sep='│')
	print('──┼──┼──')
	print(*board[3:6], sep='│')
	print('──┼──┼──')
	print(*board[6:], sep='│')

def play_game(models_x, models_o):
	for model_x in models_x:
		for model_o in models_o:
			board = ([], [], [i for i in range(9)])
			for i in range(9):
				is_x_move = i % 2 == 0
				game_board = gb.GameBoard(*board)
				x_moves, o_moves, allowed_moves = (
					game_board.x_mirrored_moves_array.tolist(), 
					game_board.o_mirrored_moves_array.tolist(), 
					game_board.allowed_mirrored_moves_array.tolist()
				)
				if (game_board.status != 'in_game'):
					break
				next_move = (model_x if is_x_move else model_o).make_move(x_moves, o_moves, is_x_move, allowed_moves)
				(x_moves if is_x_move else o_moves).append(next_move)
				allowed_moves.remove(next_move)
				board = (x_moves, o_moves, allowed_moves)

			game_board = gb.GameBoard(*board)
			x_name = model_x.config.model_name
			o_name = model_o.config.model_name
			x_wins = game_board.status == 'x_win'
			o_wins = game_board.status == 'o_win'
			draw = game_board.status == 'draw'
			# print('Крестики', x_name)
			# print('Нолики', o_name)
			# print('Игра окончена.')
			# print(game_board.status)
			# print_board(x_moves, o_moves, is_x_move)
			# print()
			if x_wins:
				with open(f'duels/{x_name}.txt', 'a+') as file:
					file.write(f'wins playing X vs {o_name} playing O\n')
				with open(f'duels/{o_name}.txt', 'a+') as file:
					file.write(f'lose playing O vs {x_name} playing X\n')
			elif o_wins:
				with open(f'duels/{x_name}.txt', 'a+') as file:
					file.write(f'lose playing X vs {o_name} playing O\n')
				with open(f'duels/{o_name}.txt', 'a+') as file:
					file.write(f'wins playing O vs {x_name} playing X\n')
			elif draw:
				with open(f'duels/{x_name}.txt', 'a+') as file:
					file.write(f'draw playing X vs {o_name} playing O\n')
				with open(f'duels/{o_name}.txt', 'a+') as file:
					file.write(f'draw playing O vs {x_name} playing X\n')
			else:
				with open(f'duels/{x_name}.txt', 'a+') as file:
					file.write(f'{x_name} playing O vs {o_name} playing X\n')
				with open(f'duels/{o_name}.txt', 'a+') as file:
					file.write(f'{o_name} playing O vs {x_name} playing X\n')

class RandomizerConfig:
	def __init__(self, index):
		self.model_name = f'!Random_{index:02d}'
		self.input_strategy_number = 0

class Randomizer:
	def __init__(self, index):
		self.config = RandomizerConfig(index)
	def make_move(self, x_moves, o_moves, is_x_move, allowed_moves):
		return random.choice(allowed_moves)
	def save_model(self, dir):
		print('Trying to save Randomizer!!')
		pass


def train(dir, model: tm.TensorflowModel, epochs, 
		  train_dataset, test_dataset):
	x_train, y_train = train_dataset
	x_test, y_test = test_dataset
	model.model.fit(
		x_train, y_train, epochs=epochs, 
		verbose=0, 
		shuffle=True,
		validation_data=(x_test, y_test), 
		validation_split=0.1)
	model.save_model(dir)
	result = model.model.evaluate(x_test, y_test, verbose=0)
	log_path = os.path.join(dir, model.config.model_name, 'train.log')
	with open(log_path, 'a') as logs:
		logs.write(str(result) + '\n')
	print(model.config.model_name, result)

def get_results(models_count, randoms_count, models):
	results = {}
	for i in range(models_count):
		if i < randoms_count:
			continue
		wins_count = 0
		lose_count = 0
		draw_count = 0
		model = models[i]
		model_name = model.config.model_name
		log_path = os.path.join('duels', model_name+'.txt')
		with open(log_path) as file:
			for line in file.readlines():
				if line.startswith('wins'):
					wins_count += 1
				elif line.startswith('lose'):
					lose_count += 1
				elif line.startswith('draw'):
					draw_count += 1
				else:
					print('err unknown result: ', line)
					continue
		total_count = sum([wins_count, draw_count, lose_count])
		results.update({i: [-lose_count, wins_count, total_count, int(wins_count/total_count * 100)]})
	return sorted(results.items(), reverse=True, key=lambda x: (x[1][0], x[1][1] / x[1][2]))

if __name__ == '__main__':
	t_models = []
	randoms_count = 5
	home_dir = 'Size_81_81_27'
	models = [Randomizer(i) for i in range(randoms_count)]
	for model_dir in [*os.walk(home_dir)][0][1]:
		if model_dir.startswith('!'):
			continue
		t_model = tm.TensorflowModel.fromFile(home_dir, model_dir)
		models.append(t_model)
		t_models.append(t_model)

	train_datasets = dict()
	test_datasets = dict()
	for i in range(1, 11):
		with open(f'datasets/train_{i:02d}.pkl', 'rb') as file:
			x_train, y_train = pickle.load(file)
			train_datasets.update({i: [x_train, y_train]})
		with open(f'datasets/test_{i:02d}.pkl', 'rb') as file:
			x_test, y_test = pickle.load(file)
			test_datasets.update({i: [x_test, y_test]})

	while True:
		models_count = len(models)
		duels_count = 5

		processes = []
		for x_i in range(models_count):
			models_set = [i for i in range(models_count) if i != x_i]
			random.shuffle(models_set)
			models_set.insert(0, x_i)
			t = Process(
				target=play_game, 
				args=(
					[models[x_i]], 
					[models[o_i] for o_i in models_set[:duels_count+1]])
			)
			processes.append(t)
		print(len(processes))
		start = time.time()
		for process in processes:
			process.start()
		for process in processes:
			process.join()
		end = time.time()
		print(f"All tournaments took {end-start:.1f} seconds")

		results = get_results(models_count, randoms_count, models)
		processes = []
		top_count = 10
		for model_i, _ in results[:top_count]:
			opponents = [models[results[i][0]] for i in range(20) if i != model_i]
			t = Process(
				target=play_game, 
				args=(
					[models[model_i]], 
					opponents
					))
			processes.append(t)
			t = Process(
				target=play_game, 
				args=(
					opponents,
					[models[model_i]]
					))
			processes.append(t)
		print(len(processes))
		start = time.time()
		for process in processes:
			process.start()
		for process in processes:
			process.join()
		end = time.time()
		print(f"Bests tournaments took {end-start:.1f} seconds")

		results = get_results(models_count, randoms_count, models)
		now = datetime.datetime.now()
		print(now)
		del_list = []
		add_list = []
		with open('training.log', 'a') as file:
			file.write(f'{now}\n')
			counter = 1
			for i, result in results:
				file.write(f'{counter:02d}. {result} {models[i].config.model_name}\n')
				counter += 1

			for i, result in results[30:]:
				deleting: tm.TensorflowModel = models[i]
				del_list.append(deleting)

			for i in range(1, 10):
				parent1: tm.TensorflowModel = models[results[0][0]]
				parent2: tm.TensorflowModel = models[results[i][0]]
				child = parent1.crossover_models(parent2)
				file.write(f'crossover 1 {child.config.model_name}\n')
				add_list.append(child)
			
			if True:
				parent1: tm.TensorflowModel = models[results[1][0]]
				parent2: tm.TensorflowModel = models[results[2][0]]
				child = parent1.crossover_models(parent2)
				file.write(f'crossover 2 {child.config.model_name}\n')
				add_list.append(child)

			for i, result in results[10:20]:
				source: tm.TensorflowModel = models[i]
				mutant = source.mutate_model()
				file.write(f'mutate add {mutant.config.model_name}\n')
				add_list.append(mutant)
			
			for i, result in results[20:30]:
				deleting: tm.TensorflowModel = models[i]
				del_list.append(deleting)
				mutant = deleting.mutate_model()
				file.write(f'mutate replace {mutant.config.model_name}\n')
				add_list.append(mutant)

		counter = 1
		for i, result in results[:3]:
			models[i].save_model(f'tournament/!top_{counter}')
			counter += 1

		now_file_postfix = f'{now.month}_{now.day}_{now.hour}_{now.minute}'
		raw_models = []
		for i in range(1, 11):
			new_name = f'New_{now_file_postfix}_{i}'
			new_config = mc.ModelConfig(
				hidden_structure=[81, 81, 27], 
				input_strategy_number=i, 
				output_strategy_number=1, 
				use_bias=True,
				use_attention=False,
				model_name=new_name
			)
			new_model = tm.TensorflowModel(new_config)
			raw_models.append(new_model)

		with open('training.log', 'a') as file:
			for deleting in del_list:
				file.write(f'removed {deleting.config.model_name}\n')
				deleting.remove_model('tournament')
				models.remove(deleting)
			file.write('\n')

		processes = []
		for model in add_list:
			strategy = model.config.input_strategy_number
			t = Process(
			target=train, 
				args=(
					'tournament', 
					model, 
					30,
					train_datasets[strategy],
					test_datasets[strategy]
					))
			processes.append(t)
		start = time.time()
		for i in range(len(processes)):
			processes[i].start()
		for i in range(len(processes)):
			processes[i].join()
		end = time.time()

		processes = []
		for model in models[randoms_count:]:
			strategy = model.config.input_strategy_number
			t = Process(
			target=train, 
				args=(
					'tournament', 
					model, 
					10,
					train_datasets[strategy],
					test_datasets[strategy]
					))
			processes.append(t)
		start = time.time()
		for i in range(len(processes)):
			processes[i].start()
		for i in range(len(processes)):
			processes[i].join()
		end = time.time()

		processes = []
		for model in raw_models:
			strategy = model.config.input_strategy_number
			t = Process(
			target=train, 
				args=(
					'tournament', 
					model, 
					100,
					train_datasets[strategy],
					test_datasets[strategy]
					))
			processes.append(t)
		start = time.time()
		for i in range(len(processes)):
			processes[i].start()
		for i in range(len(processes)):
			processes[i].join()
		end = time.time()
		print(f"All trainings took {end-start:.1f} seconds")
			
		for adding in add_list:
			models.append(adding)
		for raw_model in raw_models:
			models.append(raw_model)
		for model in models[randoms_count:]:
			model.save_model('tournament')
		shutil.copytree('tournament', f'archive/{now_file_postfix}')
		duels_path = os.path.join('duels')
		shutil.rmtree(duels_path)
		os.mkdir(duels_path)
		# break