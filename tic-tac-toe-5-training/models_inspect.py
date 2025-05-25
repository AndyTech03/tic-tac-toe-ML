import os

def get_results(models_count, randoms_count, models):
	results = {}
	for i in range(models_count):
		if i < randoms_count:
			continue
		wins_count = 0
		lose_count = 0
		draw_count = 0
		model = models[i]
		model_name = model.config.model_name
		log_path = os.path.join('duels', model_name+'.txt')
		with open(log_path) as file:
			for line in file.readlines():
				if line.startswith('wins'):
					wins_count += 1
				elif line.startswith('lose'):
					lose_count += 1
				elif line.startswith('draw'):
					draw_count += 1
				else:
					print('err unknown result: ', line)
					continue
		win_rate = wins_count / sum([wins_count, draw_count, lose_count]) * 100
		results.update({i: [-lose_count, int(win_rate), wins_count]})
	return sorted(results.items(), reverse=True, key=lambda x: (x[1][0], x[1][1]))


if __name__ == '__main__':
	import TensorflowModel as tm
	import ModelConfig as mc
	# home_dir = 'tournament'
	home_dir = 'Size_81_81_27'
	models = []
	for model_dir in [*os.walk(home_dir)][0][1]:
		if model_dir.startswith('!'):
			continue
		t_model = tm.TensorflowModel.fromFile(home_dir, model_dir)
		models.append(t_model)
	result = []
	models_count = len(models)
	randoms_count = 5
	for model in models:
		result.append((model.config.input_strategy_number, model.config.model_name))
	result.sort(key=lambda x: x[0])
	print(*result, sep='\n')
	print(get_results(models_count, randoms_count, models))
 