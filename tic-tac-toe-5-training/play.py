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

def player_move(x_moves, o_moves, player_is_x):
	print_board(x_moves, o_moves, player_is_x)
	while True:
		print('Выберите куда поставить', ('Крестик' if player_is_x else 'Нолик'))
		try:
			next_move = int(input())
		except:
			print('Ошибка')
			continue
		if next_move not in x_moves and next_move not in o_moves:
			return next_move
		else:
			print('Клетка занята', ('Крестиком' if next_move in x_moves else 'Ноликом'))

if __name__ == '__main__':
	models = []
	home_dir = 'tournament'
	for model_dir in [*os.walk(home_dir)][0][1]:
		if model_dir.startswith('!'):
			continue
		t_model = tm.TensorflowModel.fromFile(home_dir, model_dir)
		models.append(t_model)
	while True:
		t_model = None
		name = input('Введите имя модели: ')
		for model in models:
			if model.config.model_name == name:
				t_model = model
				break
		if model is None:
			print('Не найдено!', name)
			continue
		x_board = ([], [], [i for i in range(9)], True)
		o_board = ([], [], [i for i in range(9)], False)
		for j in range(2):
			player_is_x = j == 0
			game_result = 'draw'
			for i in range(9):
				is_x_move = i % 2 == 0
				x_moves, o_moves, allowed_moves, model_is_x = (
					x_board 
					if player_is_x else 
					o_board
				)
				game_board = gb.GameBoard(x_moves, o_moves, allowed_moves)
				x_moves, o_moves, allowed_moves = (
					game_board.x_mirrored_moves_array.tolist(), 
					game_board.o_mirrored_moves_array.tolist(), 
					game_board.allowed_mirrored_moves_array.tolist()
				)
				if (game_board.status != 'in_game'):
					game_result = game_board.status
					break
				next_move = (
					player_move(x_moves, o_moves, is_x_move)
					if player_is_x == is_x_move else
					model.make_move(x_moves, o_moves, model_is_x, allowed_moves)
				)
				if is_x_move:
					x_moves.append(next_move)
				else:
					o_moves.append(next_move)
				allowed_moves.remove(next_move)

				if player_is_x:
					x_board = x_moves, o_moves, allowed_moves, model_is_x
				else:
					o_board = x_moves, o_moves, allowed_moves, model_is_x
			
			x_moves, o_moves, allowed_moves, model_is_x = (
				x_board 
				if player_is_x else 
				o_board
			)
			print(game_result, 'Игрок', 'крестик' if player_is_x else 'нолик')
			print_board(x_moves, o_moves, is_x_move)
			print('Игра окончена.')