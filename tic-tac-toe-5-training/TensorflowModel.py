import pickle
import shutil
import numpy as np
import os
import importlib
from numba import njit, prange, typed
import random

# Использовать для подсветки синтаксиса
# from keras._tf_keras.keras import models, layers
# from keras._tf_keras.keras.losses import BinaryCrossentropy
# from keras._tf_keras.keras.optimizers import Adam
# Использовать для корректной работы
from tensorflow.keras import models, layers # type: ignore
from tensorflow.keras.losses import BinaryCrossentropy # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

from ModelConfig import ModelConfig, generate_model_name

class TensorflowModel:
    @staticmethod
    def fromFile(dir, model_name: str):
        config: ModelConfig = ModelConfig.load_model(dir, model_name)
        config.model_name = model_name
        t_model = TensorflowModel(config, False, dir)
        return t_model

    def __init__(self, config: ModelConfig, new=True, dir=''):
        """Создание модели на основе структуры, определённой в конфигурации."""
        self.config = config.clone()
        if new:
            self.build_model()
        else:
            self.load_model(dir)

    def increment_age(self):
        self.config.age += 1
        
    def build_model(self):
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
        self.model.compile(
            optimizer=Adam(learning_rate=1e-3, clipnorm=1.0), 
            loss=BinaryCrossentropy(), 
            metrics=['accuracy', 'mse']
            )

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

    def getModelDataset(
            self, x_train_moves, o_train_moves, 
            allowed_train_moves, model_train_answers, train_as_x, epochs=5
            ):
        if self.config.input_strategy_number == 0:
            print('Random has no needs in training.')
            return
        input_strategy, _, output_generator = self.config.get_strategies()
        
        count = len(x_train_moves)
        train_inputs = np.empty((count, self.config.get_input_structure()[0]))
        train_outputs = np.empty((count, self.config.get_output_size()))

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

        # Конвертируем в формат numpy для TensorFlow
        x_train = train_inputs
        y_train = train_outputs
        return x_train, y_train
        # Обучение модели
        # self.model.fit(x_train, y_train, epochs=epochs)
    
    def test(
            self, x_test_moves, o_test_moves, 
            allowed_test_moves, model_test_answers,
            test_as_x
            ):
        input_strategy, output_strategy, _ = self.config.get_strategies()
        correct = 0
        incorrect = 0
        count = len(x_test_moves)
        test_inputs = np.empty((count, self.config.get_input_structure()[0]))
        typed_x_train_moves = typed.List([typed.List(x) for x in x_test_moves])
        typed_o_train_moves = typed.List([typed.List(x) for x in o_test_moves])
        for i in prange(count):
            x_moves = typed_x_train_moves[i]
            o_moves = typed_o_train_moves[i]
            is_x = test_as_x[i]
            test_inputs[i, :] = input_strategy(x_moves, o_moves, is_x)
        
        output_data = self.model.predict(test_inputs)
        for i in prange(count):
            output = typed.List([typed.List(output_data[i])])
            allowed = typed.List([int(x) for x in allowed_test_moves[i]])
            output_move = output_strategy(output, allowed)
            if output_move in model_test_answers[i]:
                correct += 1
            else:
                incorrect += 1
        return correct / (correct + incorrect)
        # self.save_training_history(testing_results)

    def make_move(self, x_moves, o_moves, model_is_x, allowed_moves):
        """Выбор следующего хода используя нейросеть и выбранные стратегии ввода/вывода."""
        if len(allowed_moves) == 0:
            raise ValueError('Нет свободных ходов!')
        x_moves = np.array(x_moves)
        o_moves = np.array(o_moves)
        allowed_moves = np.array(allowed_moves)
        input_strategy, output_strategy, _ = self.config.get_strategies()
        input_data = input_strategy(x_moves, o_moves, model_is_x)
        output_data = self.model.predict(input_data, verbose=0)
        return int(output_strategy(output_data, allowed_moves))
    
    def mutate_model(self, mutation_rate: float = 0.1, mutation_scale: float = 0.05):
        """
        Создает копию модели с мутированными весами.
        :param model: Исходная модель для мутации
        :param mutation_rate: Вероятность мутации каждого параметра
        :param mutation_scale: Стандартное отклонение Гауссовского шума
        :return: Новый экземпляр TensorflowModel с мутированными весами
        """
        # Копируем конфигурацию и создаем новую модель
        strategy_changed = False
        input_strategy = self.config.input_strategy_number
        other_strategies = [i for i in range(1, 11)]
        other_strategies.remove(self.config.input_strategy_number)
        if random.random() < .032:
            strategy_changed = True
            input_strategy = random.choice(other_strategies)
        child = TensorflowModel(self.config.clone(generate_model_name(), input_strategy))
        # Получаем текущие веса
        weights = self.model.get_weights()
        child_w = child.model.get_weights()
        new_weights = []
        for w in weights:
            # Создаем маску для точек мутации
            mask = np.random.rand(*w.shape) < mutation_rate
            # Генерируем шум
            noise = np.random.normal(scale=mutation_scale, size=w.shape)
            # Применяем шум только в маске
            new_w = w + mask * noise
            new_weights.append(new_w)
        # Устанавливаем мутированные веса
        if strategy_changed:
            final_weights = []
            for cw, nw in zip(child_w, new_weights):
                if cw.shape == nw.shape:
                    final_weights.append(nw)
                else:
                    final_weights.append(cw)
            child.model.set_weights(final_weights)
        else:
            child.model.set_weights(new_weights)
        return child


    def crossover_models(parent1, parent2):
        """
        Создает нового "ребенка" путем скрещивания весов двух родительских моделей.
        Скрещивание реализовано как равновероятный отбор каждого параметра из одной из родителей.
        :param parent1: Первая родительская модель
        :param parent2: Вторая родительская модель
        :return: Новый экземпляр TensorflowModel с комбинированными весами
        """
        # Предполагаем, что у родителей одинаковая структура config
        input_strategy = parent1.config.input_strategy_number
        other_strategies = [i for i in range(1, 11)]
        other_strategies.remove(parent1.config.input_strategy_number)
        mutation_chance = .06
        if parent2.config.input_strategy_number != parent1.config.input_strategy_number:
            other_strategies.remove(parent2.config.input_strategy_number)
            mutation_chance = 0.2
        if random.random() < .4:
            if random.random() < mutation_chance:
                input_strategy = random.choice(other_strategies)
            else:
                input_strategy = parent2.config.input_strategy_number
        child = TensorflowModel(parent1.config.clone(generate_model_name(), input_strategy))
        weights1 = parent1.model.get_weights()
        weights2 = parent2.model.get_weights()
        child_w = child.model.get_weights()
        new_weights = []
        for w1, w2, target in zip(weights1, weights2, child_w):
            if w1.shape == w2.shape == target.shape:
                mask = np.random.rand(*w1.shape) < 0.5
                new_w = np.where(mask, w1, w2)
                new_weights.append(new_w)
            else:
                if parent1.config.input_strategy_number == input_strategy:
                    source = w1
                else:
                    source = w2
                # Если всё равно не подходит, оставляем инициализацию ребенка
                new_weights.append(source if source.shape == target.shape else target)

        child.model.set_weights(new_weights)
        return child

    def save_weights(self, dir):
        """Сохранение весов модели."""
        weights_path = os.path.join(dir, self.config.model_name, "model.weights.h5")
        self.model.save_weights(weights_path)
    
    def load_weights(self, dir):
        """Загрузка весов модели."""
        weights_path = os.path.join(dir, self.config.model_name, "model.weights.h5")
        if os.path.isfile(weights_path):
            self.model.load_weights(weights_path)
        else:
            print('Weights file not found!', self.config.model_name)
    
    def save_model(self, dir):
        """Сохранение полной модели в формате TensorFlow."""
        model_path = os.path.join(dir, self.config.model_name, "tensorflow_model.keras")
        self.config.save_model(dir)
        self.model.save(model_path)
        self.save_weights(dir)
    
    def load_model(self, dir):
        """Загрузка полной модели из сохранённого формата TensorFlow."""
        model_path = os.path.join(dir, self.config.model_name, "tensorflow_model.keras")
        self.model = models.load_model(model_path)
        self.load_weights(dir)

    def remove_model(self, dir):
        model_path = os.path.join(dir, self.config.model_name)
        if not os.path.isdir(model_path):
            print(f'model `{self.config.model_name}` not saved yet.')
            return
        shutil.rmtree(model_path)

    def save_score(self, dir, wins, loses, total):
        score_path = os.path.join(dir, self.config.model_name, "score.pkl")
        os.makedirs(os.path.dirname(score_path), exist_ok=True)
        score = wins, loses, total
        with open(score_path, 'wb') as file:
            pickle.dump(score, file)

    def load_score(self, dir):
        score_path = os.path.join(dir, self.config.model_name, "score.pkl")
        if not os.path.isfile(score_path):
            print(f'score `{self.config.model_name}` not saved yet.')
            return 0, 0, 0, 0
        with open(score_path, 'rb') as file:
            wins, loses, total = pickle.load(file)
            win_rate = int(wins / total * 100) if total != 0 else 0
            return loses, wins, total, win_rate