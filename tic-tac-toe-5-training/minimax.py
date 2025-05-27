from GameBoard import GameBoard
from functools import lru_cache
import numpy as np


def get_canonical_key(x_moves, o_moves):
    """
    Возвращает канонический ключ позиции через нормализацию (зеркалирование).
    """
    allowed = [i for i in range(9) if i not in x_moves and i not in o_moves]
    board = GameBoard(x_moves, o_moves, allowed_moves=allowed)
    return (tuple(board.x_mirrored_moves_array.tolist()),
            tuple(board.o_mirrored_moves_array.tolist()))


@lru_cache(maxsize=None)
def _minimax_value(x_key, o_key, is_x_turn):
    """
    Оценка позиции: +1 (X выиграл), -1 (O выиграл), 0 (ничья).
    """
    x_list = list(x_key)
    o_list = list(o_key)
    allowed = [i for i in range(9) if i not in x_list and i not in o_list]
    board = GameBoard(x_list, o_list, allowed_moves=allowed)
    status = board.status
    if status != 'in_game':
        return {'x_win': 1, 'o_win': -1}.get(status, 0)

    moves = board.allowed_mirrored_moves_array.tolist()
    if is_x_turn:
        best = -np.inf
        for mv in moves:
            kx, ko = get_canonical_key(x_list + [mv], o_list)
            val = _minimax_value(kx, ko, False)
            if val > best:
                best = val
                if best == 1:
                    break
        return best
    else:
        best = np.inf
        for mv in moves:
            kx, ko = get_canonical_key(x_list, o_list + [mv])
            val = _minimax_value(kx, ko, True)
            if val < best:
                best = val
                if best == -1:
                    break
        return best


def minimax(x_moves, o_moves, model_is_x=True):
    """
    Выбирает оптимальный ход для модели (X или O).
    model_is_x=True для X, False для O.
    """
    # Получаем нормализованные ключи
    x_key, o_key = get_canonical_key(tuple(x_moves), tuple(o_moves))
    # Создаем доску для allowed_moves
    allowed = GameBoard(x_moves, o_moves,
                        allowed_moves=[i for i in range(9) if i not in x_moves + o_moves]
                       ).allowed_mirrored_moves_array.tolist()

    best_mv = None
    if model_is_x:
        best_score = -np.inf
        for mv in allowed:
            kx, ko = get_canonical_key(x_moves + [mv], o_moves)
            score = _minimax_value(kx, ko, False)
            if score > best_score:
                best_score, best_mv = score, mv
                if best_score == 1:
                    break
    else:
        best_score = np.inf
        for mv in allowed:
            kx, ko = get_canonical_key(x_moves, o_moves + [mv])
            score = _minimax_value(kx, ko, True)
            if score < best_score:
                best_score, best_mv = score, mv
                if best_score == -1:
                    break
    return best_score, best_mv

class MinimaxConfig:
	def __init__(self, index):
		self.model_name = f'!Minimax_{index:02d}'
		self.input_strategy_number = 0

class MinimaxModel:
    def __init__(self, index):
        self.config = MinimaxConfig(index)
    def make_move(self, x_moves, o_moves, model_is_x, allowed_moves):
        score, move = minimax(x_moves, o_moves, model_is_x)
        return move
    def save_model(self, dir):
        print('Trying to save Randomizer!!')

if __name__ == '__main__':
    # Пример использования
    x_moves = [0]
    o_moves = [1]
    score, move = minimax(x_moves, o_moves, model_is_x=True)
    print(f"Best opening move for X: {move}, score: {score}")