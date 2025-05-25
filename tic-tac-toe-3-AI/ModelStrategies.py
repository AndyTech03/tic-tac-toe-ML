import numpy as np
from numpy import ndarray
from numba import prange, njit
# from ModelConfig import ModelConfig

simple_19_size = 19#ModelConfig.get_input_structure_by_number(1)
simple_19x3_size = 55#ModelConfig.get_input_structure_by_number(2)
simple_19x5_size = 91#ModelConfig.get_input_structure_by_number(4)


analytic_17_size = 17#ModelConfig.get_input_structure_by_number(6)
analytic_17x3_size = 49#ModelConfig.get_input_structure_by_number(7)
analytic_17x5_size = 81#ModelConfig.get_input_structure_by_number(9)

@njit
def count_in(items: ndarray, collection: ndarray):
    count = len(items)
    counter = 0
    for i in prange(count):
        if items[i] in collection:
            counter += 1
    if counter == 0 or count == 0:
        return 0.01
    if counter == count:
        return 0.99
    return counter / count

class InputStrategies:
    @staticmethod
    @njit
    def random(x_moves, o_moves, model_is_x):
        return np.empty((1, 1))
    
    @staticmethod
    @njit
    def simple_19(x_moves, o_moves, model_is_x):
        inputs: ndarray = np.empty((1, simple_19_size))
        for i in prange(9):
            inputs[0, i] = 0.99 if i in x_moves else 0.01
            inputs[0, i+9] = 0.99 if i in o_moves else 0.01
        inputs[0, 18] = 0.99 if model_is_x else 0.01
        return inputs
    
    @staticmethod
    @njit
    def simple_19x3_from_start(x_moves, o_moves, model_is_x):
        inputs = np.empty((1, simple_19x3_size))
        x_count = len(x_moves)
        o_count = len(o_moves)
        start_x = 0 if 3 >= x_count else x_count - 3
        start_o = 0 if 3 >= o_count else o_count - 3
        for layer in prange(3):
            for i in prange(9):
                if layer >= x_count:
                    inputs[0, layer*9 + i] = 0.01
                else:
                    inputs[0, layer*9 + i] = 0.99 if i in x_moves[:start_x+layer+1] else 0.01
                if layer >= x_count:
                    inputs[0, layer*9 + i + 9*3] = 0.01
                else:
                    inputs[0, layer*9 + i + 9*3] = 0.99 if i in o_moves[:start_o+layer+1] else 0.01
        inputs[0, simple_19x3_size-1] = 0.99 if model_is_x else 0.01
        return inputs
    
    @staticmethod
    @njit
    def simple_19x3_from_end(x_moves, o_moves, model_is_x):
        inputs = np.empty((1, simple_19x3_size))
        x_count = len(x_moves)
        o_count = len(o_moves)
        for layer in prange(3):
            for i in prange(9):
                if layer >= x_count:
                    inputs[0, layer*9 + i] = 0.01
                else:
                    inputs[0, layer*9 + i] = 0.99 if i in x_moves[:x_count-layer] else 0.01
                if layer >= x_count:
                    inputs[0, layer*9 + i + 9*3] = 0.01
                else:
                    inputs[0, layer*9 + i + 9*3] = 0.99 if i in o_moves[:o_count-layer] else 0.01
        inputs[0, simple_19x3_size-1] = 0.99 if model_is_x else 0.01
        return inputs
    
    @staticmethod
    @njit
    def simple_19x5_from_start(x_moves, o_moves, model_is_x):
        inputs = np.empty((1, simple_19x5_size))
        x_count = len(x_moves)
        o_count = len(o_moves)
        start_x = 0 if 5 >= x_count else x_count - 5
        start_o = 0 if 5 >= o_count else o_count - 5
        for layer in prange(5):
            for i in prange(9):
                if layer >= x_count:
                    inputs[0, layer*9 + i] = 0.01
                else:
                    inputs[0, layer*9 + i] = 0.99 if i in x_moves[:start_x+layer+1] else 0.01
                
                if layer >= o_count:
                    inputs[0, layer*9 + i + 5*9] = 0.01
                else:
                    inputs[0, layer*9 + i + 5*9] = 0.99 if i in o_moves[:start_o+layer+1] else 0.01
                
        inputs[0, simple_19x5_size-1] = 0.99 if model_is_x else 0.01
        return inputs

    @staticmethod
    @njit
    def simple_19x5_from_end(x_moves, o_moves, model_is_x):
        inputs = np.empty((1, simple_19x5_size))
        x_count = len(x_moves)
        o_count = len(o_moves)
        for layer in prange(5):
            for i in prange(9):
                if layer >= x_count:
                    inputs[0, layer*9 + i] = 0.01
                else:
                    inputs[0, layer*9 + i] = 0.99 if i in x_moves[:x_count-layer] else 0.01
                if layer >= x_count:
                    inputs[0, layer*9 + i + 9*5] = 0.01
                else:
                    inputs[0, layer*9 + i + 9*5] = 0.99 if i in o_moves[:o_count-layer] else 0.01
        inputs[0, simple_19x5_size-1] = 0.99 if model_is_x else 0.01
        return inputs
    

    @staticmethod
    @njit
    def analytic_17(x_moves, o_moves, model_is_x):
        inputs = np.empty((1, analytic_17_size))
        # 3 Горизонтали X
        inputs[0, 0] = count_in([0, 1, 2], x_moves)
        inputs[0, 1] = count_in([3, 4, 5], x_moves)
        inputs[0, 2] = count_in([6, 7, 8], x_moves)

        # 3 Вертикали X
        inputs[0, 3] = count_in([0, 3, 6], x_moves)
        inputs[0, 4] = count_in([1, 4, 7], x_moves)
        inputs[0, 5] = count_in([2, 5, 8], x_moves)

        # 2 Диагонали X
        inputs[0, 6] = count_in([0, 4, 8], x_moves)
        inputs[0, 7] = count_in([2, 4, 6], x_moves)
        
        # 3 Горизонтали O
        inputs[0, 8] = count_in([0, 1, 2], o_moves)
        inputs[0, 9] = count_in([3, 4, 5], o_moves)
        inputs[0, 10] = count_in([6, 7, 8], o_moves)

        # 3 Вертикали O
        inputs[0, 11] = count_in([0, 3, 6], o_moves)
        inputs[0, 12] = count_in([1, 4, 7], o_moves)
        inputs[0, 13] = count_in([2, 5, 8], o_moves)

        # 2 Диагонали O
        inputs[0, 14] = count_in([0, 4, 8], o_moves)
        inputs[0, 15] = count_in([2, 4, 6], o_moves)

        inputs[0, 16] = 0.99 if model_is_x else 0.01
        return inputs
    
    @staticmethod
    @njit
    def analytic_17x3_from_start(x_moves, o_moves, model_is_x):
        inputs = np.empty((1, analytic_17x3_size))
        x_count = len(x_moves)
        o_count = len(o_moves)
        start_x = 0 if 3 >= x_count else x_count - 3
        start_o = 0 if 3 >= o_count else o_count - 3
        for layer in prange(3):
            if layer >= x_count:
                for i in range(8):
                    inputs[0, layer*8 + i] = 0.01
            else:
                # 3 Горизонтали X
                inputs[0, layer*8 + 0] = count_in([0, 1, 2], x_moves[:start_x+layer+1])
                inputs[0, layer*8 + 1] = count_in([3, 4, 5], x_moves[:start_x+layer+1])
                inputs[0, layer*8 + 2] = count_in([6, 7, 8], x_moves[:start_x+layer+1])

                # 3 Вертикали X
                inputs[0, layer*8 + 3] = count_in([0, 3, 6], x_moves[:start_x+layer+1])
                inputs[0, layer*8 + 4] = count_in([1, 4, 7], x_moves[:start_x+layer+1])
                inputs[0, layer*8 + 5] = count_in([2, 5, 8], x_moves[:start_x+layer+1])

                # 2 Диагонали X
                inputs[0, layer*8 + 6] = count_in([0, 4, 8], x_moves[:start_x+layer+1])
                inputs[0, layer*8 + 7] = count_in([2, 4, 6], x_moves[:start_x+layer+1])

            if layer >= o_count:
                for i in range(8):
                    inputs[0, layer*8 + i + 3*8] = 0.01
            else:
                # 3 Горизонтали O
                inputs[0, layer*8 + 0 + 3*8] = count_in([0, 1, 2], o_moves[:start_o+layer+1])
                inputs[0, layer*8 + 1 + 3*8] = count_in([3, 4, 5], o_moves[:start_o+layer+1])
                inputs[0, layer*8 + 2 + 3*8] = count_in([6, 7, 8], o_moves[:start_o+layer+1])

                # 3 Вертикали O
                inputs[0, layer*8 + 3 + 3*8] = count_in([0, 3, 6], o_moves[:start_o+layer+1])
                inputs[0, layer*8 + 4 + 3*8] = count_in([1, 4, 7], o_moves[:start_o+layer+1])
                inputs[0, layer*8 + 5 + 3*8] = count_in([2, 5, 8], o_moves[:start_o+layer+1])

                # 2 Диагонали O
                inputs[0, layer*8 + 6 + 3*8] = count_in([0, 4, 8], o_moves[:start_o+layer+1])
                inputs[0, layer*8 + 7 + 3*8] = count_in([2, 4, 6], o_moves[:start_o+layer+1])

        inputs[0, analytic_17x3_size-1] = 0.99 if model_is_x else 0.01
        return inputs

    @staticmethod
    @njit
    def analytic_17x3_from_end(x_moves, o_moves, model_is_x):
        inputs = np.empty((1, analytic_17x3_size))
        x_count = len(x_moves)
        o_count = len(o_moves)
        for layer in prange(3):
            if layer >= x_count:
                for i in range(8):
                    inputs[0, layer*8 + i] = 0.01
            else:
                # 3 Горизонтали X
                inputs[0, layer*8 + 0] = count_in([0, 1, 2], x_moves[:x_count-layer])
                inputs[0, layer*8 + 1] = count_in([3, 4, 5], x_moves[:x_count-layer])
                inputs[0, layer*8 + 2] = count_in([6, 7, 8], x_moves[:x_count-layer])

                # 3 Вертикали X
                inputs[0, layer*8 + 3] = count_in([0, 3, 6], x_moves[:x_count-layer])
                inputs[0, layer*8 + 4] = count_in([1, 4, 7], x_moves[:x_count-layer])
                inputs[0, layer*8 + 5] = count_in([2, 5, 8], x_moves[:x_count-layer])

                # 2 Диагонали X
                inputs[0, layer*8 + 6] = count_in([0, 4, 8], x_moves[:x_count-layer])
                inputs[0, layer*8 + 7] = count_in([2, 4, 6], x_moves[:x_count-layer])

            if layer >= o_count:
                for i in range(8):
                    inputs[0, layer*8 + i + 3*8] = 0.01
            else:
                # 3 Горизонтали O
                inputs[0, layer*8 + 0 + 3*8] = count_in([0, 1, 2], o_moves[:o_count-layer])
                inputs[0, layer*8 + 1 + 3*8] = count_in([3, 4, 5], o_moves[:o_count-layer])
                inputs[0, layer*8 + 2 + 3*8] = count_in([6, 7, 8], o_moves[:o_count-layer])

                # 3 Вертикали O
                inputs[0, layer*8 + 3 + 3*8] = count_in([0, 3, 6], o_moves[:o_count-layer])
                inputs[0, layer*8 + 4 + 3*8] = count_in([1, 4, 7], o_moves[:o_count-layer])
                inputs[0, layer*8 + 5 + 3*8] = count_in([2, 5, 8], o_moves[:o_count-layer])

                # 2 Диагонали O
                inputs[0, layer*8 + 6 + 3*8] = count_in([0, 4, 8], o_moves[:o_count-layer])
                inputs[0, layer*8 + 7 + 3*8] = count_in([2, 4, 6], o_moves[:o_count-layer])

        inputs[0, analytic_17x3_size-1] = 0.99 if model_is_x else 0.01
        return inputs
    
    @staticmethod
    @njit
    def analytic_17x5_from_start(x_moves, o_moves, model_is_x):
        inputs = np.empty((1, analytic_17x5_size))
        x_count = len(x_moves)
        o_count = len(o_moves)
        start_x = 0 if 5 >= x_count else x_count - 5
        start_o = 0 if 5 >= o_count else o_count - 5
        for layer in prange(5):
            if layer >= x_count:
                for i in range(8):
                    inputs[0, layer*8 + i] = 0.01
            else:
                # 3 Горизонтали X
                inputs[0, layer*8 + 0] = count_in([0, 1, 2], x_moves[:start_x+layer+1])
                inputs[0, layer*8 + 1] = count_in([3, 4, 5], x_moves[:start_x+layer+1])
                inputs[0, layer*8 + 2] = count_in([6, 7, 8], x_moves[:start_x+layer+1])

                # 3 Вертикали X
                inputs[0, layer*8 + 3] = count_in([0, 3, 6], x_moves[:start_x+layer+1])
                inputs[0, layer*8 + 4] = count_in([1, 4, 7], x_moves[:start_x+layer+1])
                inputs[0, layer*8 + 5] = count_in([2, 5, 8], x_moves[:start_x+layer+1])

                # 2 Диагонали X
                inputs[0, layer*8 + 6] = count_in([0, 4, 8], x_moves[:start_x+layer+1])
                inputs[0, layer*8 + 7] = count_in([2, 4, 6], x_moves[:start_x+layer+1])
            
            if layer >= x_count:
                for i in range(8):
                    inputs[0, layer*8 + i + 5*8] = 0.01
            else:
                # 3 Горизонтали O
                inputs[0, layer*8 + 0 + 5*8] = count_in([0, 1, 2], o_moves[:start_o+layer+1])
                inputs[0, layer*8 + 1 + 5*8] = count_in([3, 4, 5], o_moves[:start_o+layer+1])
                inputs[0, layer*8 + 2 + 5*8] = count_in([6, 7, 8], o_moves[:start_o+layer+1])

                # 3 Вертикали O
                inputs[0, layer*8 + 3 + 5*8] = count_in([0, 3, 6], o_moves[:start_o+layer+1])
                inputs[0, layer*8 + 4 + 5*8] = count_in([1, 4, 7], o_moves[:start_o+layer+1])
                inputs[0, layer*8 + 5 + 5*8] = count_in([2, 5, 8], o_moves[:start_o+layer+1])

                # 2 Диагонали O
                inputs[0, layer*8 + 6 + 5*8] = count_in([0, 4, 8], o_moves[:start_o+layer+1])
                inputs[0, layer*8 + 7 + 5*8] = count_in([2, 4, 6], o_moves[:start_o+layer+1])

        inputs[0, analytic_17x5_size-1] = 0.99 if model_is_x else 0.01
        return inputs
    
    @staticmethod
    @njit
    def analytic_17x5_from_end(x_moves, o_moves, model_is_x):
        inputs = np.empty((1, analytic_17x5_size))
        x_count = len(x_moves)
        o_count = len(o_moves)
        start_x = 0 if 5 >= x_count else x_count - 5
        start_o = 0 if 5 >= o_count else o_count - 5
        for layer in prange(5):
            if layer >= x_count:
                for i in range(8):
                    inputs[0, layer*8 + i] = 0.01
            else:
                # 3 Горизонтали X
                inputs[0, layer*8 + 0] = count_in([0, 1, 2], x_moves[:x_count-layer])
                inputs[0, layer*8 + 1] = count_in([3, 4, 5], x_moves[:x_count-layer])
                inputs[0, layer*8 + 2] = count_in([6, 7, 8], x_moves[:x_count-layer])

                # 3 Вертикали X
                inputs[0, layer*8 + 3] = count_in([0, 3, 6], x_moves[:x_count-layer])
                inputs[0, layer*8 + 4] = count_in([1, 4, 7], x_moves[:x_count-layer])
                inputs[0, layer*8 + 5] = count_in([2, 5, 8], x_moves[:x_count-layer])

                # 2 Диагонали X
                inputs[0, layer*8 + 6] = count_in([0, 4, 8], x_moves[:x_count-layer])
                inputs[0, layer*8 + 7] = count_in([2, 4, 6], x_moves[:x_count-layer])
            
            if layer >= x_count:
                for i in range(8):
                    inputs[0, layer*8 + i + 5*8] = 0.01
            else:
                # 3 Горизонтали O
                inputs[0, layer*8 + 0 + 5*8] = count_in([0, 1, 2], o_moves[:o_count-layer])
                inputs[0, layer*8 + 1 + 5*8] = count_in([3, 4, 5], o_moves[:o_count-layer])
                inputs[0, layer*8 + 2 + 5*8] = count_in([6, 7, 8], o_moves[:o_count-layer])

                # 3 Вертикали O
                inputs[0, layer*8 + 3 + 5*8] = count_in([0, 3, 6], o_moves[:o_count-layer])
                inputs[0, layer*8 + 4 + 5*8] = count_in([1, 4, 7], o_moves[:o_count-layer])
                inputs[0, layer*8 + 5 + 5*8] = count_in([2, 5, 8], o_moves[:o_count-layer])

                # 2 Диагонали O
                inputs[0, layer*8 + 6 + 5*8] = count_in([0, 4, 8], o_moves[:o_count-layer])
                inputs[0, layer*8 + 7 + 5*8] = count_in([2, 4, 6], o_moves[:o_count-layer])

        inputs[0, analytic_17x5_size-1] = 0.99 if model_is_x else 0.01
        return inputs

class OutputStrategies:
    @staticmethod
    @njit
    def random(model_output: ndarray, available_moves):
        return np.random.choice(np.array(available_moves))
    
    @staticmethod
    @njit
    def simple_9(model_output: ndarray, available_moves: ndarray):
        length = len(available_moves)
        max_available_output = None
        max_available_moves = np.empty((length))
        count = 0
        for i in prange(length):
            move = available_moves[i]
            output = model_output[0, move]
            # Если сигнал выше максимального, заменяем
            if (max_available_output is None or
                    output > max_available_output):
                max_available_output = output
                count = 1
                # max_available_moves[:] = np.nan # Не обязательно
                max_available_moves[0] = move
                continue
            # Если сингал совпадает, добавляем
            if output == max_available_output:
                max_available_moves[count] = move
                count += 1
        # Выбираем случайный из массива максимальных
        return np.random.choice(max_available_moves[:count])

            

class OutputGenerators:
    @staticmethod
    @njit
    def random(selected_moves, allowed_moves):
        return np.zeros((1, 1))
    
    @staticmethod
    # @njit
    def simple_9(selected_moves, allowed_moves):
        perfect_outputs = np.empty((9,))
        for i in prange(9):
            # Правильный ответ сети
            if i in selected_moves:
                perfect_outputs[i] = .99
                continue
            # Не правильный ответ сети
            if i in allowed_moves:
                perfect_outputs[i] = .33
                continue
            # Недопустимый ответ сети
            perfect_outputs[i] = .01#-.99
        return perfect_outputs
