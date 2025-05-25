import os
import pickle
import numpy as np
from numpy import ndarray
from numba import prange, njit
from faker import Faker
from typing_extensions import Self
from ModelStrategies import InputStrategies, OutputStrategies, OutputGenerators

def get_local_directories():
    """Возвращает список директорий, расположенных рядом со скриптом."""
    script_path = os.path.dirname(os.path.abspath(__file__))
    return [entry.name for entry in os.scandir(script_path) if entry.is_dir()]

def generate_model_name():
    fake = Faker()
    dirs = get_local_directories()
    while True:
        random_name = fake.first_name()
        unique_id = fake.random_int(min=1, max=1000)
        model_name = f"{random_name}_Model_{unique_id}"
        if model_name not in dirs:
            return model_name

class ModelConfig:
    def __init__(
            self, hidden_structure: list[int],
            input_strategy_number: int, output_strategy_number: int,
            use_bias: bool=False, use_attention: bool=False, model_name: str=None
            ):
        self.model_name = model_name if model_name is not None else generate_model_name()
        self.hidden_structure = hidden_structure
        self.input_strategy_number = input_strategy_number
        self.output_strategy_number = output_strategy_number
        self.use_attention = use_attention
        self.use_bias = use_bias

        for size in hidden_structure:
            if size <= 0:
                raise ValueError('Размер должен быть больше 0!')

        os.makedirs(self.model_name, exist_ok=True)
        # self.generate_random_weights()
        self.save_model()
    
    @staticmethod
    def get_input_structure_by_number(input_strategy_number: int):
        """Возвращает структуру входного слоя в зависимости от полученного номера."""
        # 2 поля по 9 позиций
        simple_input = 9 * 2
        # 2 поля по 3 вертикали, 3 горизонтали и 2 диагонали
        analytic_input = (3 + 3 + 2) * 2

        # Минимальный размер
        if input_strategy_number == 0:
            return (1, )
        
        # simple_input + 1 символ
        elif input_strategy_number == 1:
            return (simple_input + 1, )
        
        # simple_input * 3 + 1 символ
        elif input_strategy_number == 2:
            return (3 * simple_input + 1, )
        elif input_strategy_number == 3:
            return (3 * simple_input + 1, )
        # simple_input * 5 + 1 символ
        elif input_strategy_number == 4:
            return (5 * simple_input + 1, )
        elif input_strategy_number == 5:
            return (5 * simple_input + 1, )
        
        # analytic_input + 1 символ
        elif input_strategy_number == 6:
            return (analytic_input + 1, )
        # analytic_input * 3 + 1 символ
        elif input_strategy_number == 7:
            return (3 * analytic_input + 1, )
        elif input_strategy_number == 8:
            return (3 * analytic_input + 1, )
        # analytic_input * 5 + 1 символ
        elif input_strategy_number == 9:
            return (5 * analytic_input + 1, )
        elif input_strategy_number == 10:
            return (5 * analytic_input + 1, )
        
        raise ValueError(f'Стратегия входного слоя №{input_strategy_number} не реализована!')

    def get_input_structure(self: Self):
        return ModelConfig.get_input_structure_by_number(self.input_strategy_number)
        

    def get_output_size_by_number(output_strategy_number: int):
        """Возвращает размер выходного слоя в зависимости от выбранной стратегии."""
        # Успешность (для текущей позиции)
        if output_strategy_number == 0:
            return 1
        # Успешность (для каждого хода)
        elif output_strategy_number == 1:
            return 9
        
        raise ValueError(f'Стратегия выходного слоя №{output_strategy_number} не реализована!')
    
    def get_output_size(self: Self):
        """Возвращает размер выходного слоя в зависимости от от полученного номера."""
        return ModelConfig.get_output_size_by_number(self.output_strategy_number)
        
    @njit
    def generate_random_weights(self):
        """Генерация весов случайных весов по заданной структуре.
        И сохранение их в файл 'weights.npy'."""
        max_size = max(self.structure)
        count = len(self.structure)
        weights = np.empty((count, max_size, max_size))
        for i in prange(1, count):
            previous_structure = self.structure[i - 1]
            layer_structure = self.structure[i]
            # Заполняем действительные веса случайными значениями
            weights[
                i, :previous_structure, :layer_structure
            ] = np.random.random((previous_structure, layer_structure))
            # Заполняем остальное np.nan
            weights[i, previous_structure:, :] = np.nan
            weights[i, :, layer_structure:] = np.nan
        weights[0, :] = np.nan
        self.save_weights(weights)
    
    def get_strategies(self):
        input_strategy = None
        output_strategy = None
        output_generator = None
        if self.input_strategy_number == 0:
            input_strategy = InputStrategies.random

        elif self.input_strategy_number == 1:
            input_strategy = InputStrategies.simple_19
        elif self.input_strategy_number == 2:
            input_strategy = InputStrategies.simple_19x3_from_start
        elif self.input_strategy_number == 3:
            input_strategy = InputStrategies.simple_19x3_from_end
        elif self.input_strategy_number == 4:
            input_strategy = InputStrategies.simple_19x5_from_start
        elif self.input_strategy_number == 5:
            input_strategy = InputStrategies.simple_19x5_from_end

        elif self.input_strategy_number == 6:
            input_strategy = InputStrategies.analytic_17
        elif self.input_strategy_number == 7:
            input_strategy = InputStrategies.analytic_17x3_from_start
        elif self.input_strategy_number == 8:
            input_strategy = InputStrategies.analytic_17x3_from_end
        elif self.input_strategy_number == 9:
            input_strategy = InputStrategies.analytic_17x5_from_start
        elif self.input_strategy_number == 10:
            input_strategy = InputStrategies.analytic_17x5_from_end

        if self.output_strategy_number == 0:
            output_strategy = OutputStrategies.random
            output_generator = OutputGenerators.random
        if self.output_strategy_number == 1:
            output_strategy = OutputStrategies.simple_9
            output_generator = OutputGenerators.simple_9
        
        return input_strategy, output_strategy, output_generator
    
    def save_model(self):
        """Сохранение модели в файл 'model.pkl'"""
        model_filepath = os.path.join(self.model_name, "model.pkl")
        with open(model_filepath, 'wb') as file:
            pickle.dump(self, file)
    
    @staticmethod
    def load_model(model_name):
        model_filepath = os.path.join(model_name, "model.pkl")
        with open(model_filepath, 'rb') as file:
            return pickle.load(file)
    
     # Сохранение весов
    def save_weights(self, weights_data: list[list]):
        """Сохранение весов в файл 'weights.npy'."""
        self.save_info('weights', weights_data)

    # Загрузка весов
    def load_weights(self) -> list[list]:
        """Загрузка весов из файла 'weights.npy'."""
        return self.load_info('weights')

    # Сохранение заметок
    def save_notes(self, notes_title, notes_data):
        """Сохранение заметок в файл с заголовком notes_title."""
        notes_filepath = os.path.join(self.model_name, f"{notes_title}.txt")
        with open(notes_filepath, 'w') as file:
            file.write(notes_data)

    # Загрузка заметок
    def load_notes(self, notes_title):
        """Загрузка заметок из файла с заголовком notes_title."""
        notes_filepath = os.path.join(self.model_name, f"{notes_title}.txt")
        with open(notes_filepath, 'r') as file:
            return file.read()

    # Сохранение NDArray
    def save_ndarray(self, array_title, array_data):
        """Сохранение NDArray в файл с заголовком array_title."""
        array_filepath = os.path.join(self.model_name, f"{array_title}.npy")
        np.save(array_filepath, array_data)

    # Загрузка NDArray
    def load_ndarray(self, array_title):
        """Загрузка NDArray из файла с заголовком array_title."""
        array_filepath = os.path.join(self.model_name, f"{array_title}.npy")
        return np.load(array_filepath)

    # Сохранение информации
    def save_info(self, info_title, info_data):
        """Сохранение информации в файл с заголовком info_title."""
        info_filepath = os.path.join(self.model_name, f"{info_title}.pkl")
        with open(info_filepath, 'wb') as file:
            pickle.dump(info_data, file)

    # Загрузка информации
    def load_info(self, info_title):
        """Загрузка информации из файла с заголовком info_title."""
        info_filepath = os.path.join(self.model_name, f"{info_title}.pkl")
        with open(info_filepath, 'rb') as file:
            return pickle.load(file)

    # Сохранение результатов
    def save_results(self, results_title, results_data):
        """Сохранение результатов в файл с заголовком results_title."""
        results_filepath = os.path.join(self.model_name, f"{results_title}.pkl")
        with open(results_filepath, 'wb') as file:
            pickle.dump(results_data, file)

    # Загрузка результатов
    def load_results(self, results_title):
        """Загрузка результатов из файла с заголовком results_title."""
        results_filepath = os.path.join(self.model_name, f"{results_title}.pkl")
        with open(results_filepath, 'rb') as file:
            return pickle.load(file)

    # Запись лога
    def write_log(self, log_title, log_message):
        """Добавление сообщения в лог в файл с заголовком log_title."""
        log_filepath = os.path.join(self.model_name, f"{log_title}.txt")
        with open(log_filepath, 'a') as file:
            file.write(log_message + '\n')

