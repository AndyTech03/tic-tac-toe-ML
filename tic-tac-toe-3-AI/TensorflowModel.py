import numpy as np
import os
import importlib
from numba import njit, prange, typed

# Использовать для подсветки синтаксиса
from keras._tf_keras.keras import models, layers
# Использовать для корректной работы
# from tensorflow.keras import models, layers # type: ignore

from ModelConfig import ModelConfig

class TensorflowModel:
    def __init__(self, config: ModelConfig):
        """Создание модели на основе структуры, определённой в конфигурации."""
        self.config = config
        self.model = models.Sequential()
        # Входные слои
        input_structure = self.config.get_input_structure()
        self.model.add(layers.Input(shape=input_structure))
        
        # Скрытые слои
        for layer_size in self.config.hidden_structure:
            self.model.add(layers.Dense(layer_size, activation='relu', use_bias=self.config.use_bias))
        
        # Слой внимания
        if self.config.use_attention:
            self.model.add(layers.AdditiveAttention())

        # Выходные слои
        output_size = self.config.get_output_size()
        activation = 'softmax' if  output_size != 1 else 'sigmoid'
        self.model.add(layers.Dense(output_size, activation=activation, use_bias=self.config.use_bias))

        # Компиляция модели
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.save_weights()
        # Загрузка весов модели
        self.load_weights()
    
    @staticmethod
    # @njit()
    def get_train_data(
            train_as_x, train_inputs, train_outputs, 
            input_strategy, output_generator,
            x_train_moves, o_train_moves,
            model_train_answers, allowed_train_moves 
            ):
        count = len(x_train_moves)
        for i in prange(count):
            train_inputs[i, :] = input_strategy(x_train_moves[i], o_train_moves[i], train_as_x[i])
            train_outputs[i, :] = output_generator(model_train_answers[i], allowed_train_moves[i])

    def train(
            self, x_train_moves, o_train_moves, 
            allowed_train_moves, model_train_answers, train_as_x
            ):
        if self.config.input_strategy_number == 0:
            print('Random has no needs in training.')
            return
        input_strategy, _, output_generator = self.config.get_strategies()
        
        count = len(x_train_moves)
        train_inputs = np.empty((count, self.config.get_input_structure()[0]))
        train_outputs = np.empty((count, self.config.get_output_size()))
        print(train_inputs, train_outputs)

        typed_x_train_moves = typed.List([typed.List(x) for x in x_train_moves])
        typed_o_train_moves = typed.List([typed.List(x) for x in o_train_moves])
        typed_model_train_answers = typed.List([typed.List(x) for x in model_train_answers])
        typed_allowed_train_moves = typed.List([typed.List(x) for x in allowed_train_moves])
        self.get_train_data(
            train_as_x, train_inputs, train_outputs,
            input_strategy, output_generator, 
            typed_x_train_moves, typed_o_train_moves,
            typed_model_train_answers, typed_allowed_train_moves
        )
        print(train_inputs, train_outputs)

        # Конвертируем в формат numpy для TensorFlow
        x_train = train_inputs
        y_train = train_outputs
        
        # Обучение модели
        self.model.fit(x_train, y_train, epochs=20)
    
    def test(
            self, x_test_moves, o_test_moves, 
            allowed_test_moves, model_test_answers
            ):
        testing_results = ([], [])
        for figure in prange(2):
            test_as_x = figure == 0
            count = len(x_test_moves[figure])
            for i in prange(count):
                model_move = self.make_move(
                    x_test_moves[figure][i], o_test_moves[figure][i],
                    test_as_x, allowed_test_moves[figure][i]
                )
                testing_results[figure].append(model_move in model_test_answers[figure][i])
        self.save_training_history(testing_results)

    def make_move(self, x_moves, o_moves, model_is_x, allowed_moves):
        """Выбор следующего хода используя нейросеть и выбранные стратегии ввода/вывода."""
        if len(allowed_moves) == 0:
            raise ValueError('Нет свободных ходов!')
        input_strategy, output_strategy, _ = self.config.get_strategies()
        input_data = input_strategy(x_moves, o_moves, model_is_x)
        print(input_data)
        output_data = self.model.predict(input_data)
        print(output_data)
        return output_strategy(output_data, allowed_moves)
    
    def save_weights(self):
        """Сохранение весов модели."""
        weights = self.model.get_weights()
        self.config.save_weights(weights)
    
    def load_weights(self):
        """Загрузка весов модели."""
        weights = self.config.load_weights()
        self.model.set_weights(weights)
    
    def save_model(self):
        """Сохранение полной модели в формате TensorFlow."""
        model_path = os.path.join(self.config.model_name, "tensorflow_model.keras")
        self.model.save(model_path)
    
    def load_model(self):
        """Загрузка полной модели из сохранённого формата TensorFlow."""
        model_path = os.path.join(self.config.model_name, "tensorflow_model.keras")
        self.model = models.load_model(model_path)
    
    def save_training_history(self, history, title="training_history"):
        """Сохранение истории обучения."""
        self.config.save_info(title, history)
    
    def load_training_history(self, title="training_history"):
        """Загрузка истории обучения."""
        return self.config.load_info(title)
