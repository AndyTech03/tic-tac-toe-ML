import numpy as np
from numba import prange, njit, typed

from dataset_generator import generate_datasets


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

class Datasets:
	def __init__(self):
		self.endings = self.read_endings()
		self.lines = [
				[0, 1, 2], [3, 4, 5], [6, 7, 8],
				[0, 3, 6], [1, 4, 7], [2, 5, 8],
				[0, 4, 8], [2, 4, 6],
			]
		
	@staticmethod
	def generate_train_dataset():
		# x_wins_boards = np.genfromtxt('x_win_boards.csv', delimiter=',', skip_header=1, dtype=np.int8)
		# o_wins_boards = np.genfromtxt('o_win_boards.csv', delimiter=',', skip_header=1, dtype=np.int8)
		# draws_boards = np.genfromtxt('draws_boards.csv', delimiter=',', skip_header=1, dtype=np.int8)
		in_game_boards = np.genfromtxt('in_game_boards.csv', delimiter=',', skip_header=1, dtype=np.int64)
		# x_wins_boards, o_wins_boards, draws_boards,

		init = GameBoard.init
		measure_axes = GameBoard.measure_axes
		delta_axes = GameBoard.delta_axes
		flip_left_top = GameBoard.flip_left_top
		flip_left = GameBoard.flip_left
		flip_left_down = GameBoard.flip_left_down
		flip_down = GameBoard.flip_down
		@njit
		def init_game_board(x_moves_array, o_moves_array, ignore_last_x, ignore_last_o):
			init(x_moves_array, o_moves_array,
				measure_axes, delta_axes, flip_left_top, 
				flip_left, flip_left_down, flip_down, ignore_last_x, ignore_last_o
			)
		with (
			open('x_train_dataset.csv', 'w') as x_train_dataset, 
			open('o_train_dataset.csv', 'w') as o_train_dataset
			):
			header_line = 'x,1,2,3,4,5,o,1,2,3,4,L,[Moves]\n'
			x_train_dataset.write(header_line)
			o_train_dataset.write(header_line)

			boards_count = len(in_game_boards)
			skip_count = 0
			pass_count = 0
			x_train_count = 0
			o_train_count = 0
			to_x_fork_count = 0
			to_o_fork_count = 0
			for board_i in prange(boards_count):
				board = in_game_boards[board_i]
				x_train = []
				o_train = []
				skipped = False
				fork = False
				for test in generate_datasets(board, GameBoard.get_status):
					if test[0] == 0:
						x_train.append(test[1])
					elif test[0] == 1:
						o_train.append(test[1])
					elif test[0] == -1:
						skip_count += 1
						skipped = True
						# break
					elif test[0] == -2:
						fork = True
						if test[1] == 0:
							to_x_fork_count += 1
						elif test[1] == 1:
							to_o_fork_count += 1
						else:
							fork = False
				count_x = len(x_train)
				count_o = len(o_train)
				if not (skipped or fork) and count_x == 0 and count_o == 0:
					pass_count += 1
					continue
				if count_x > 0:
					line = ','.join(str(x) for x in board) + f',{count_x},[' + ' '.join(str(x) for x in x_train) + ']\n'
					x_train_dataset.write(line)
					x_train_count += 1
				if count_o > 0:
					line = ','.join(str(x) for x in board) + f',{count_o},[' + ' '.join(str(x) for x in o_train) + ']\n'
					o_train_dataset.write(line)
					o_train_count += 1

			print('boards_count:', boards_count)
			print('skip_count:', skip_count)
			print('pass_count:', pass_count)
			print('x_train_count:', x_train_count)
			print('o_train_count:', o_train_count)
			print('to_x_fork_count:', to_x_fork_count)
			print('to_o_fork_count:', to_o_fork_count)


	@staticmethod
	@njit
	def get_train_dataset(in_game_boards: np.ndarray, get_status, init_game_board):
		'''
		1. Использует результаты Datasets.generate_boards (449 247 позиций из 986 409 возможных - 55% оптимизация)
		2. Будут использоваться только незавершенные "in_game: (101 475 позиций)
		3. Из которых извлекаем такие, что однозначно завершаются победой или ничьей (16 264 позиций)
		4. Добавляем позиции когда игрок ставить "вилку" (5 592 позиций)
		5. Так же добавляем обход "вилок" (5 592 позиций)
		6. Итого 27 448 позиций для обучения из 101 475 - 27% покрытие
		'''
		counter_1 = 0
		counter_2 = 0

		boards_count = len(in_game_boards)
		for board_i in prange(boards_count):
			board: np.ndarray = in_game_boards[board_i]
			x_count: int = board[0]
			o_count: int = board[6]
			delta = x_count - o_count
			next_move = 'x' if delta == 0 else 'o'
			x_moves_array = board[1:1+x_count]
			o_moves_array = board[7:7+o_count]
			allowed_moves_array = np.array([
				i for i in prange(9) 
				if  i not in x_moves_array and
					i not in o_moves_array], dtype=np.int8)
			# print(x_moves_array, o_moves_array, allowed_moves_array, next_move, delta)
			allowed_count = len(allowed_moves_array)
			move_status_array = np.empty((allowed_count,), dtype=np.str_)
			any_win = False 	# хотя бы один возможный ход привёл к победе
			any_in_game = False	# хотя бы один возможный ход не завершает игру
			# Определяем статусы для всех возможных ходов
			for move_i in prange(allowed_count):
				move = allowed_moves_array[move_i]
				x_moves_test = x_moves_array.copy()
				o_moves_test = o_moves_array.copy()
				if next_move == 'x':
					x_moves_test = np.append(x_moves_test, move)
				elif next_move == 'o':
					o_moves_test = np.append(o_moves_test, move)
				status = get_status(x_moves_test, o_moves_test)
				move_status_array[move_i] = status
				if status == 'x_win' or status == 'o_win':
					any_win = True
				elif status == 'in_game':
					any_in_game = True

			is_block_moves = False	#
			positive_moves = np.empty((0, ), dtype=np.int8)
			for move_i in prange(allowed_count):
				move = allowed_moves_array[move_i]
				status = move_status_array[move_i]
				# print(x_moves_test, o_moves_test, move, status)
				if status == 'draw':
					# если Ничья, то это последний ход, и других нет
					positive_moves = np.array([move])
					# break	
				elif any_win and (status == 'x_win' or status == 'o_win'):
					# если Победа, то добавляем этот ход, игнорируя ходы, которые затянут игру
					positive_moves = np.append(positive_moves, [move])
				elif any_win == False and any_in_game:
					# если победных ходов нет, Ходим в любое другое место, а текущий ход "передаём" оппоненту
					other_move = allowed_moves_array[allowed_moves_array != move][0]
					x_moves_test = None
					o_moves_test = None
					if next_move == 'x':
						x_moves_test = np.append(x_moves_array, other_move)
						o_moves_test = np.append(o_moves_array, move)
					elif next_move == 'o':
						x_moves_test = np.append(x_moves_array, move)
						o_moves_test = np.append(o_moves_array, other_move)
					# если он побеждает, мы обязаны блокировать этот ход
					other_status = get_status(x_moves_test, o_moves_test)
					if other_status == 'o_win' or other_status == 'x_win':
						positive_moves = np.append(positive_moves, [move])
						is_block_moves = True

			positive_moves_count = len(positive_moves)
			if positive_moves_count == 0:
				continue

			if is_block_moves and positive_moves_count > 1:
				# Противник сделал вилку, предыдущий ход был ошибкой
				x_moves_test = x_moves_array.copy()
				o_moves_test = o_moves_array.copy()
				wrong_move = 9
				# строим позицию, на которой был совершён неверный ход, ignore_last_x = True, ignore_last_o = True
				init_game_board(x_moves_test, o_moves_test, True, True)
				x_i = len(x_moves_test) - 1
				o_i = len(o_moves_test) - 1
				# Отменяем последние 2 хода: ход противника, ошибочный ход
				if next_move == 'x':
					wrong_move = x_moves_test[x_i]
					x_moves_test = x_moves_test[:x_i]
					o_moves_test = o_moves_test[:o_i]
				elif next_move == 'o':
					wrong_move = o_moves_test[o_i]
					x_moves_test = x_moves_test[:x_i]
					o_moves_test = o_moves_test[:o_i]
				
				x_test_count = len(x_moves_test)
				o_test_count = len(o_moves_test)
				x_moves_test_board = np.array([
						x_moves_test[i] if i <= x_test_count - 1 else 9 for i in prange(5)
					], dtype=np.int8)
				o_moves_test_board = np.array([
						o_moves_test[i] if i <= o_test_count - 1 else 9 for i in prange(4)
					], dtype=np.int8)
				test_board = np.array([
						x_test_count, *x_moves_test_board, 
						o_test_count, *o_moves_test_board
					], dtype=np.int8)
				
				# добавляем этот ход в обучение как неправильный
				train_marker = 0
				args = np.array([delta, train_marker, wrong_move], dtype=np.int8)
				yield np.append(test_board, args)

			if is_block_moves and positive_moves_count > 1:
				# Противник сделал вилку, а значит этот ход победный
				x_moves_test = x_moves_array.copy()
				o_moves_test = o_moves_array.copy()
				positive_move = 9
				# Отменяем последние ход противника
				if next_move == 'x':
					# строим позицию, на которой была вилка, ignore_last_x = False, ignore_last_o = True
					init_game_board(x_moves_test, o_moves_test, False, True)
					o_i = len(o_moves_test) - 1
					positive_move = o_moves_test[o_i]
					o_moves_test = o_moves_test[:o_i]
				elif next_move == 'o':
					# строим позицию, на которой была вилка, ignore_last_x = True, ignore_last_o = False
					init_game_board(x_moves_test, o_moves_test, True, False)
					x_i = len(x_moves_test) - 1
					positive_move = o_moves_test[x_i]
					x_moves_test = x_moves_test[:x_i]
				
				x_test_count = len(x_moves_test)
				o_test_count = len(o_moves_test)
				x_moves_test_board = np.array([
						x_moves_test[i] if i <= x_test_count - 1 else 9 for i in prange(5)
					], dtype=np.int8)
				o_moves_test_board = np.array([
						o_moves_test[i] if i <= o_test_count - 1 else 9 for i in prange(4)
					], dtype=np.int8)
				test_board = np.array([
						x_test_count, *x_moves_test_board, 
						o_test_count, *o_moves_test_board
					], dtype=np.int8)
				
				# добавляем этот ход в обучение как правильный
				train_marker = 1
				test_delta = 0 if delta == 1 else 1
				args = np.array([test_delta, train_marker, positive_move], dtype=np.int8)
				yield np.append(test_board, args)
				counter_2 += 1
				
			# добавляем ходы из positive_moves в обучение как правильные
			train_marker = 1
			args = np.array([delta, train_marker, *positive_moves], dtype=np.int8)
			yield np.append(board, args)
			counter_1 += 1
		print()
		print('pure wins =', counter_1, ', twin wins =', counter_2)

	@staticmethod
	def generate_boards():
		'''
		1. Вызывает "Datasets.generator (986 409 позиций)". 
		2. Затем оптимизирует их через GameBoard (568 407 позиций).
		3. После разделяет по полю "status", отсекая невозможные (449 247 позиций).
		'''
		boards = Datasets.generator()
		for board in boards[:16]:
			print(board)
		print('....')
		for board in boards[-16:]:
			print(board)

		unique_boards = set()
		count = len(boards)
		invalid_count = 0
		in_game_counter = 0
		with (
			open('all_positions.csv', 'w') as all_positions,
			open('unique_positions.csv', 'w') as unique_positions,
			open('x_win_boards.csv', 'w') as x_wins, 
			open('o_win_boards.csv', 'w') as o_wins, 
			open('draws_boards.csv', 'w') as draws,
			open('in_game_boards.csv', 'w') as in_game,
			):
			header_line = 'x,1,2,3,4,5,o,1,2,3,4\n'
			all_positions.write(header_line)
			unique_positions.write(header_line)
			x_wins.write(header_line)
			o_wins.write(header_line)
			draws.write(header_line)
			in_game.write(header_line)

			for board_i in prange(count):
				x_count = boards[board_i, 5]
				o_count = boards[board_i, 5+1+4]
				x_moves = boards[board_i, :x_count]
				o_moves = boards[board_i, 6:6+o_count]
				# print(x_moves, o_moves, boards[board_i])
				b = GameBoard(x_moves, o_moves)
				all_positions.write(b.get_csv_line())

				# GameBoard.print(b.x_moves_array, b.o_moves_array)
				count = len(unique_boards)
				unique_boards.add(b)
				print('\r', board_i, end='                                 ')

				if count == len(unique_boards):
					continue

				unique_positions.write(b.get_csv_line())
				status = b.status
				scv_line = b.get_csv_line()
				if status == 'in_game':
					in_game_counter += 1
				if status == 'x_win':
					x_wins.write(scv_line)
				elif status == 'o_win':
					o_wins.write(scv_line)
				elif status == 'draw':
					draws.write(scv_line)
				elif status == 'in_game':
					in_game.write(scv_line)
				else:
					invalid_count += 1

		print()
		print(len(boards))
		print(len(unique_boards))
		print(invalid_count)
		print(in_game_counter)

	def encode_endings(self, source):
		'''
		Функция кодировки данных из дата-сета числами
		'b' = 0
		'x' = 1
		'o' = 2
		'neutral' = 3
		'positive' = 4
		'negative' = 5
		'''
		encoded_type = '10u1'
		endings_count = len(source)
		result = np.empty(endings_count, dtype=encoded_type)
		for i in range(endings_count):
			ending = source[i]
			temp = []
			for figure in ending[:9]:
				if figure == 'b':
					temp.append(0)
				elif figure == 'x':
					temp.append(1)
				elif figure == 'o':
					temp.append(2)

			if ending[9] == 'neutral':
				temp.append(3)
			elif ending[9] == 'positive':
				temp.append(4)
			elif ending[9] == 'negative':
				temp.append(5)
			result[i] = temp
		return result

	def read_endings(self):
		'''
		Функция чтения дата-сета
		'''
		source = np.genfromtxt('tic-tac-toe.data', delimiter=',', dtype='U30')
		for ending in source:
			if ending[9] == 'negative' and 'b' not in ending:
				ending[9] = 'neutral'
		self.source = source
		return self.encode_endings(source)

	@staticmethod
	def print(x_moves, o_moves):
		for i in prange(9):
			if i in x_moves:
				print('x', end=' ')
			elif i in o_moves:
				print('o', end=' ')
			else:
				print('b', end=' ')
			if (i+1) % 3 == 0:
				print()
		print()

	def get_operated_boards(self):
		count = len(self.source)
		boards = set()
		print(count)
		for ending_i in prange(count):
			ending = self.source[ending_i]
			x_moves = []
			o_moves = []
			for i in prange(9):
				figure = ending[i]
				if figure == 'x':
					x_moves.append(i)
				elif figure == 'o':
					o_moves.append(i)
			b = GameBoard(x_moves, o_moves)
			boards.add(b)
		print(len(boards))

	
	@staticmethod
	@njit
	def generator():
		'''Генерирует 986 409 уникальных позиций, учитывая порядок ходов.'''
		map_0 = 0
		map_1 = 1 * (map_0 + 1)
		map_2 = 2 * (map_1 + 1)
		map_3 = 3 * (map_2 + 1)
		map_4 = 4 * (map_3 + 1)
		map_5 = 5 * (map_4 + 1)
		map_6 = 6 * (map_5 + 1)
		map_7 = 7 * (map_6 + 1)
		map_8 = 8 * (map_7 + 1)
		map_9 = 9 * (map_8 + 1)
		
		boards = np.zeros((map_9, 5+1 + 4+1), dtype=np.int8)
		x_count_i = 5
		o_start_i = x_count_i+1
		o_count_i = o_start_i+4
		for x1 in prange(9):
			x1_start = x1*(map_8 + 1)
			x1_end = x1_start + map_8 + 1
			boards[x1_start:x1_end, 0] = x1
			boards[x1_start:x1_end, x_count_i] = 1

			o1_i = 0
			for o1 in prange(9):
				if o1 in {x1}:
					continue
				o1_start = x1_start + 1 + o1_i*(map_7 + 1)
				o1_end = o1_start + map_7 + 1 
				boards[o1_start:o1_end, o_start_i + 0] = o1
				boards[o1_start:o1_end, o_count_i] = 1
				o1_i += 1
			
				x2_i = 0
				for x2 in prange(9):
					if x2 in {x1, o1}:
						continue
					x2_start = o1_start + 1 + x2_i*(map_6 + 1)
					x2_end = x2_start + map_6 + 1
					boards[x2_start:x2_end, 1] = x2
					boards[x2_start:x2_end, x_count_i] = 2
					x2_i += 1
				
					o2_i = 0
					for o2 in prange(9):
						if o2 in {x1, o1, x2}:
							continue
						o2_start = x2_start + 1 + o2_i*(map_5 + 1)
						o2_end = o2_start + map_5 + 1
						boards[o2_start:o2_end, o_start_i + 1] = o2
						boards[o2_start:o2_end, o_count_i] = 2
						o2_i += 1

						x3_i = 0
						for x3 in prange(9):
							if x3 in {x1, o1, x2, o2}:
								continue
							x3_start = o2_start + 1 + x3_i*(map_4 + 1)
							x3_end = x3_start + map_4 + 1
							boards[x3_start:x3_end, 2] = x3
							boards[x3_start:x3_end, x_count_i] = 3
							x3_i += 1

							o3_i = 0
							for o3 in prange(9):
								if o3 in {x1, o1, x2, o2, x3}:
									continue
								o3_start = x3_start + 1 + o3_i*(map_3 + 1)
								o3_end = o3_start + map_3 + 1
								boards[o3_start:o3_end, o_start_i + 2] = o3
								boards[o3_start:o3_end, o_count_i] = 3
								o3_i += 1

								x4_i = 0
								for x4 in prange(9):
									if x4 in {x1, o1, x2, o2, x3, o3}:
										continue
									x4_start = o3_start + 1 + x4_i*(map_2 + 1)
									x4_end = x4_start + map_2 + 1
									boards[x4_start:x4_end, 3] = x4
									boards[x4_start:x4_end, x_count_i] = 4
									x4_i += 1

									o4_i = 0
									for o4 in prange(9):
										if o4 in {x1, o1, x2, o2, x3, o3, x4}:
											continue
										o4_start = x4_start + 1 + o4_i*(map_1 + 1)
										o4_end = o4_start + map_1 + 1
										boards[o4_start:o4_end, o_start_i + 3] = o4
										boards[o4_start:o4_end, o_count_i] = 4
										o4_i += 1

										x5_i = 0
										for x5 in prange(9):
											if x5 in {x1, o1, x2, o2, x3, o3, x4, o4}:
												continue
											x5_start = o4_start + 1 + x5_i*(map_0 + 1)
											x5_end = x5_start + map_0 + 1
											boards[x5_start:x5_end, 4] = x5
											boards[x5_start:x5_end, x_count_i] = 5
											x5_i += 1
		return boards

class GameBoard:
	def __init__(self, x_moves, o_moves):
		self.x_moves_array = np.array(x_moves, dtype=np.int8)
		self.o_moves_array = np.array(o_moves, dtype=np.int8)
		self.x_mirrored_moves_array = np.array(x_moves, dtype=np.int8)
		self.o_mirrored_moves_array = np.array(o_moves, dtype=np.int8)
		self.status = GameBoard.get_status(self.x_moves_array, self.o_moves_array)

		GameBoard.init(self.x_mirrored_moves_array, self.o_mirrored_moves_array,
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
	def init(x_moves_array, o_moves_array, 
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
			for array in [x_moves_array, o_moves_array]:
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

if __name__ == '__main__':
	Datasets.generate_boards()
	Datasets.generate_train_dataset()
	# dataset = np.genfromtxt('train_dataset.csv', delimiter=',', skip_header=1, dtype=np.str_)
	# for x in dataset:
	# 	print(x)