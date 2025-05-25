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
import GameBoard as gb
# importlib.reload(gb)

def get_test_data(situation):
		x_moves = []
		o_moves = []
		allowed_moves = [
			0, 1, 2,
			3, 4, 5,
			6, 7, 8,
		]
		index = -1
		for row in situation.split('\n'):
			for figure in row.split(' '):
				index += 1
				if figure == 'b':
					continue
				if figure == 'x':
					x_moves.append(index)
					allowed_moves.remove(index)
					continue
				if figure == 'o':
					o_moves.append(index)
					allowed_moves.remove(index)
		is_x = len(allowed_moves) % 2 == 1
		
		return x_moves, o_moves, allowed_moves, is_x

if __name__ == '__main__':
	models = []
	processes = []
	for i in range(1, 11):
		model_name = f'Size_100_100/Inputs_{i:02d}'
		config = mc.ModelConfig(
			hidden_structure=[100, 100], 
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
			raise FileNotFoundError()
		models.append(t_model)

	while True:
		print('Введите карту:')
		situation = input('1. ') + '\n'
		situation += input('2. ') + '\n'
		situation += input('3. ')
		x_moves, o_moves, allowed_moves, model_is_x = get_test_data(situation)
		game_board = gb.GameBoard(x_moves, o_moves, allowed_moves)
		x_moves, o_moves, allowed_moves = game_board.x_mirrored_moves_array, game_board.o_mirrored_moves_array, game_board.allowed_mirrored_moves_array
		print(x_moves, o_moves, allowed_moves, model_is_x)
		board = [
			'  ', '  ', '  ',
			'  ', '  ', '  ',
			'  ', '  ', '  ',
		]
		for i in x_moves:
			board[i] = '><'
		for i in o_moves:
			board[i] = '()'
		results = dict()
		for model in models:
			move = model.make_move(x_moves, o_moves, model_is_x, allowed_moves)
			if move in results:
				results[move].append(model.config.model_name)
			else:
				results.update({ move: [model.config.model_name] })
		results = dict(sorted(results.items(), key=lambda x: len(x[1]), reverse=True))
		for move in results:
			move_board = [c for c in board]
			move_board[move] = '}{' if model_is_x else '[]'
			print(*results[move], sep=', ')
			print(*move_board[:3], sep='│')
			print('──┼──┼──')
			print(*move_board[3:6], sep='│')
			print('──┼──┼──')
			print(*move_board[6:], sep='│')
			print()
