import numpy as np
from numba import prange, njit, typed

@njit
def all_in(items, collection):
    count = len(items)
    for i in prange(count):
        if items[i] not in collection:
            return False
    return True

@njit
def count_in(items, collection):
    counter = 0
    count = len(items)
    for i in prange(count):
        if items[i] in collection:
            counter += 1
    return counter

class GameBoard:
	def __init__(self, x_moves, o_moves, allowed_moves=[], correct_moves=[]):
		self.x_moves_array = np.array(x_moves, dtype=np.int8)
		self.o_moves_array = np.array(o_moves, dtype=np.int8)
		self.allowed_moves = np.array(allowed_moves, dtype=np.int8)
		self.correct_moves = np.array(correct_moves, dtype=np.int8)
		self.x_mirrored_moves_array = np.array(x_moves, dtype=np.int8)
		self.o_mirrored_moves_array = np.array(o_moves, dtype=np.int8)
		self.allowed_mirrored_moves_array = np.array(allowed_moves, dtype=np.int8)
		self.correct_mirrored_moves_array = np.array(correct_moves, dtype=np.int8)
		self.status = GameBoard.get_status(self.x_moves_array, self.o_moves_array)

		GameBoard.init(
			self.x_mirrored_moves_array, self.o_mirrored_moves_array, 
			self.allowed_mirrored_moves_array, self.correct_mirrored_moves_array,
			GameBoard.measure_axes, GameBoard.delta_axes, GameBoard.flip_left_top, 
			GameBoard.flip_left, GameBoard.flip_left_down, GameBoard.flip_down
		)

	def get_csv_line(self):
		csv = ''
		x_count = len(self.x_moves_array)
		o_count = len(self.o_moves_array)
		x_moves = [str(x_count)] + [str(x) for x in self.x_moves_array] + ['9']*(5-x_count)
		o_moves = [str(o_count)] + [str(o) for o in self.o_moves_array] + ['9']*(4-o_count)
		csv += ','.join(x_moves + o_moves)
		return csv + '\n'

	def get_mirrored_csv_line(self):
		csv = ''
		x_count = len(self.x_mirrored_moves_array)
		o_count = len(self.o_mirrored_moves_array)
		x_moves = [str(x_count)] + [str(x) for x in self.x_mirrored_moves_array] + ['9']*(5-x_count)
		o_moves = [str(o_count)] + [str(o) for o in self.o_mirrored_moves_array] + ['9']*(4-o_count)
		csv += ','.join(x_moves + o_moves)
		return csv + '\n'

	@staticmethod
	@njit
	def init(x_moves_array, o_moves_array, allowed_mirrored_moves_array, correct_mirrored_moves_array,
			measure_axes, delta_axes, flip_left_top, flip_left, flip_left_down, flip_down, 
			ignore_last_x=False, ignore_last_o=False):
		left_top = [[0, 1, 3], [5, 7, 8]]
		left = [[0, 3, 6], [2, 5, 8]]
		left_down = [[3, 6, 7], [1, 2, 5]]
		down = [[6, 7, 8], [0, 1, 2]]
		left_vs_down = [[0, 3, 6], [6, 7, 8]]
		axes_array = np.array([left_top, left, left_down, down, left_vs_down])
		# print(x_moves_array, o_moves_array)
		# GameBoard.print(x_moves_array, o_moves_array)
		while True:
			measured_axes = measure_axes(axes_array, x_moves_array, o_moves_array, ignore_last_x, ignore_last_o)
			axes_delta = delta_axes(measured_axes)
			err_axis_i = np.argmin(
				axes_delta, axis=None
			)
			if axes_delta[err_axis_i] >= 0:
				break
			for array in [x_moves_array, o_moves_array, allowed_mirrored_moves_array, correct_mirrored_moves_array]:
				if err_axis_i == 0:
					flip_left_top(array)
				elif err_axis_i == 1:
					flip_left(array)
				elif err_axis_i == 2:
					flip_left_down(array)
				elif err_axis_i == 3:
					flip_down(array)
				elif err_axis_i == 4:
					flip_left_top(array)

	@staticmethod
	@njit
	def flip_left_top(array):
		change_map = [
			8, 5, 2,
			7, 4, 1,
			6, 3, 0,
		]
		for i in prange(len(array)):
			array[i] = change_map[array[i]]

	@staticmethod
	@njit
	def flip_left(array):
		change_map = [
			2, 1, 0,
			5, 4, 3,
			8, 7, 6,
		]
		for i in prange(len(array)):
			array[i] = change_map[array[i]]
	
	@staticmethod
	@njit
	def flip_left_down(array):
		change_map = [
			0, 3, 6,
			1, 4, 7,
			2, 5, 8,
		]
		for i in prange(len(array)):
			array[i] = change_map[array[i]]

	@staticmethod
	@njit
	def flip_down(array):
		change_map = [
			6, 7, 8,
			3, 4, 5,
			0, 1, 2,
		]
		for i in prange(len(array)):
			array[i] = change_map[array[i]]

	@staticmethod
	@njit
	def measure_axes(axes_array, x_moves_array: np.ndarray, o_moves_array: np.ndarray, ignore_last_x=False, ignore_last_o=False):
		axes_power_array = np.empty((4, 2), dtype=np.int8)
		measure_x_moves = x_moves_array.copy()
		measure_o_moves = o_moves_array.copy()
		# если указано ignore_last_moves не учитываем последние ходы игроков
		if ignore_last_x:
			x_count = len(measure_x_moves)
			measure_x_moves = measure_x_moves[:x_count-1]
		if ignore_last_o:
			o_count = len(measure_o_moves)
			measure_o_moves = measure_o_moves[:o_count-1]
		all_moves = np.concatenate((measure_x_moves, measure_o_moves))

		for axis_i in prange(len(axes_array)):
			axis = axes_array[axis_i]
			for vector_i in prange(len(axis)):
				vector = axis[vector_i]
				axes_power_array[axis_i, vector_i] = count_in(vector, all_moves)
		return axes_power_array

	@staticmethod
	@njit
	def delta_axes(measured_axes):
		axes_delta_array = np.empty((4,), dtype=np.int8)
		for axis_i in prange(len(measured_axes)):
			axis = measured_axes[axis_i]
			axes_delta_array[axis_i] = axis[0] - axis[1]
		return axes_delta_array

	@staticmethod
	@njit
	def get_status(x_moves_array, o_moves_array):
		lines = [
			[0, 1, 2], [3, 4, 5], [6, 7, 8],
			[0, 3, 6], [1, 4, 7], [2, 5, 8],
			[0, 4, 8], [2, 4, 6],
		]
		x_win = False
		o_win = False
		for line in lines:
			if all_in(line, x_moves_array):
				x_win = True
				break
		for line in lines:
			if all_in(line, o_moves_array):
				o_win = True
				break
		
		x_count = len(x_moves_array)
		o_count = len(o_moves_array)
		delta = x_count - o_count

		if count_in(x_moves_array, o_moves_array) != 0:
			return 'overlapping'
		elif delta > 1:
			return 'too_many_x_moves'
		elif delta < 0:
			return 'too_many_o_moves'
		elif x_win and o_win:
			return 'double_win'
		elif x_win:
			return 'x_win'
		elif o_win:
			return 'o_win'
		elif x_count + o_count == 9:
			return 'draw'
		else:
			return 'in_game'
	
	@staticmethod
	def print(x_moves_array, o_moves_array):
		for i in prange(9):
			if i in x_moves_array:
				print('x', end=' ')
			elif i in o_moves_array:
				print('o', end=' ')
			else:
				print('b', end=' ')
			if (i+1) % 3 == 0:
				print()
		print()
	
	def __hash__(self) -> int:
		hash_str = ''
		for x in self.x_mirrored_moves_array:
			hash_str += str(x)
		hash_str += '|'
		for o in self.o_mirrored_moves_array:
			hash_str += str(o)
		return hash(hash_str)
	
	def __eq__(self, other: object) -> bool:
		return hash(self) == hash(other)
	
	def __str__(self) -> str:
		hash_str = ''
		for x in self.x_moves_array:
			hash_str += str(x)
		hash_str += '|'
		for o in self.o_moves_array:
			hash_str += str(o)
		return hash_str
	
	def __repr__(self) -> str:
		return str(self)
