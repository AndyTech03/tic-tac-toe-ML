import numpy as np
from numba import prange, njit, typed


@njit
def get_board(x_moves, o_moves):
	x_count = len(x_moves)
	o_count = len(o_moves)
	return np.array([
		x_count, *x_moves, *([9] * (5 - x_count)), 
		o_count, *o_moves, *([9] * (4 - o_count)), 
	], dtype=np.int8)

@njit
def get_x_moves(board):
	x_count: int = board[0]
	return board[1:1+x_count]

@njit
def get_o_moves(board):
	o_count: int = board[6]
	o_moves = board[7:7+o_count]
	return o_moves

@njit
def get_next_moves(board: np.ndarray):
	x_moves_array = get_x_moves(board)
	o_moves_array = get_o_moves(board)
	x_count, o_count = len(x_moves_array), len(o_moves_array)
	delta = x_count - o_count
	figure = 'x' if delta == 0 else 'o'
	allowed_moves_array = np.array([
		i for i in prange(9) 
		if  i not in x_moves_array and
			i not in o_moves_array], dtype=np.int8)
	allowed_count = len(allowed_moves_array)
	# Переберем следующие ходы
	for move_i in prange(allowed_count):
		move = allowed_moves_array[move_i]
		x_moves_test = x_moves_array.copy()
		o_moves_test = o_moves_array.copy()
		if figure == 'x':
			x_moves_test = np.append(x_moves_test, move)
		elif figure == 'o':
			o_moves_test = np.append(o_moves_test, move)
		yield np.array([*get_board(x_moves_test, o_moves_test), delta, move])

@njit
def generate_datasets(board: np.ndarray, get_status):
	x_moves_array = get_x_moves(board)
	o_moves_array = get_o_moves(board)
	x_count, o_count = len(x_moves_array), len(o_moves_array)
	delta = x_count - o_count
	analyse_boards = np.empty((0, 11, ), dtype=np.int64)
	in_game_moves = np.empty((0, ), dtype=np.int64)
	moves = [item for item in get_next_moves(board)]
	moves_count = len(moves)
	any_win = False
	# Если остался один ход - учить нечему
	if moves_count == 1:
		yield [-1, -9]		
		return
	for move_i in prange(moves_count):
		move_data = moves[move_i]
		move_board = move_data[:11]
		move_delta = int(move_data[11])
		move = int(move_data[12])
		figure = 'x' if move_delta == 0 else 'o'
		x_moves = get_x_moves(move_board)
		o_moves = get_o_moves(move_board)
		status = get_status(x_moves, o_moves)
		# Победный ход выходим
		if status == f'{figure}_win':
			any_win = True
			# Значит обучаем delta игрока выбирать move
			yield [delta, move]
		elif status == 'in_game':
			in_game_moves = np.append(in_game_moves, [move])
			move_board = move_board.reshape((1, 11))
			analyse_boards = np.concatenate((analyse_boards, move_board))
		else:
			print('Unexpected status', status, 'was skipped.')

	# Есть победный ход - выходим
	if any_win:
		return

	# Иначе партия продолжается - проверяем следующие ходы
	second_defend_moves = np.empty((0, ), dtype=np.int64)
	for first_move_i in prange(len(in_game_moves)):
		second_board = analyse_boards[first_move_i]
		for defend_move in second_move_analyse(second_board, get_status):
			if defend_move in second_defend_moves:
				continue
			second_defend_moves = np.append(second_defend_moves, [defend_move])
	defend_moves_count = len(second_defend_moves)
	if defend_moves_count == 1:
		# Значит обучаем delta игрока выбирать second_defend_moves[0]
		yield [delta, second_defend_moves[0]]
	if defend_moves_count > 1:
		yield [-2, delta]

@njit
def second_move_analyse(second_board, get_status):
	for move_data in get_next_moves(second_board):
		move_board = move_data[:11]
		move_delta = int(move_data[11])
		second_move = int(move_data[12])
		figure = 'x' if move_delta == 0 else 'o'
		x_moves = get_x_moves(move_board)
		o_moves = get_o_moves(move_board)
		status = get_status(x_moves, o_moves)
		if status == f'{figure}_win':
			yield second_move

