import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

from GameBoard import GameBoard


def simulate_branch(x_moves, o_moves, allowed_moves, model, model_is_x):
    """
    Explore game tree from given state: model plays on its turn, opponent enumerates all moves.
    Returns (stats, logs) where stats is Counter({'total', 'wins', 'losses', 'draws'})
    and logs is list of (x_moves, o_moves, model_is_x, status).
    """
    stats = Counter()
    logs = []

    def dfs(x, o, allowed):
        board = GameBoard(x, o)
        status = board.status
        if status != 'in_game':
            stats['total'] += 1
            # determine win/loss/draw for model
            win = (status == 'x_win' and model_is_x) or (status == 'o_win' and not model_is_x)
            loss = (status == 'o_win' and model_is_x) or (status == 'x_win' and not model_is_x)
            if win:
                stats['wins'] += 1
                # logs.append((x.copy(), o.copy(), model_is_x, status))
            elif loss:
                stats['losses'] += 1
                logs.append((x.copy(), o.copy(), model_is_x, status))
            else:
                stats['draws'] += 1
            return

        # determine whose turn
        move_count = len(x) + len(o)
        is_x_move = (move_count % 2 == 0)
        is_model_turn = (is_x_move == model_is_x)

        if is_model_turn:
            # model chooses a single move
            mv = model.make_move(x, o, model_is_x, allowed)
            x_new, o_new = (x + [mv], o) if is_x_move else (x, o + [mv])
            allowed_new = [m for m in allowed if m != mv]
            dfs(x_new, o_new, allowed_new)
        else:
            # opponent enumerates all moves
            for mv in list(allowed):
                x_new, o_new = (x + [mv], o) if is_x_move else (x, o + [mv])
                allowed_new = [m for m in allowed if m != mv]
                dfs(x_new, o_new, allowed_new)

    dfs(list(x_moves), list(o_moves), list(allowed_moves))
    return stats, logs


def test_model(model, model_is_x, max_workers, logs=True):
    """
    Test the given TensorflowModel playing as X if model_is_x True, else as O.
    Uses multithreading to parallelize over initial opponent moves.
    """
    # initial empty board
    init_x, init_o = [], []
    init_allowed = list(range(9))
    # determine first branching tasks
    # if opponent moves first, branch on each allowed; if model first, one task
    first_turn_is_model = (model_is_x and True) or (not model_is_x and False)
    tasks = []
    if first_turn_is_model:
        tasks.append((init_x, init_o, init_allowed))
    else:
        # opponent is X and starts
        for mv in init_allowed:
            x1 = [mv]
            o1 = []
            allowed1 = [m for m in init_allowed if m != mv]
            tasks.append((x1, o1, allowed1))

    total_stats = Counter()
    total_logs = []
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(simulate_branch, x, o, allowed, model, model_is_x): (x, o) for x, o, allowed in tasks}
        for future in as_completed(futures):
            stats, logs = future.result()
            with lock:
                total_stats.update(stats)
                if logs:
                    total_logs.extend(logs)

    return total_stats, total_logs

def test_models(dir, models):
    def test_and_update(dir, model):
        counter_x, _ = test_model(model, True, 9, False)
        counter_o, _ = test_model(model, False, 9, False)
        loses = counter_x['losses'] + counter_o['losses']
        wins = counter_x['wins'] + counter_o['wins']
        total = counter_x['total'] + counter_o['total']
        model.save_score(dir, wins, loses, total)
    for model in models:
        test_and_update(dir, model)

# def main():
#     parser = argparse.ArgumentParser(description='Test TensorflowModel outcomes over entire game tree.')
#     parser.add_argument('model_dir', help='Directory where model is saved')
#     parser.add_argument('models_names', help='Names of the models')
#     parser.add_argument('--workers', type=int, default=8, help='Number of threads')
#     args = parser.parse_args()

#     # t_model = MinimaxModel(1)
#     # model_name = 'MinimaxModel'
#     for model_name in args.models_names.split(','):
#         # load model twice for X and O
#         # t_model = TensorflowModel.fromFile(args.model_dir, model_name)
#     print(f"Testing model '{model_name}' as X...")
#     stats_x, logs_x = test_model(t_model, True, args.workers)
#     print(f"As X: total={stats_x['total']}, wins={stats_x['wins']}, draws={stats_x['draws']}, losses={stats_x['losses']}")
#     stats_o, logs_o = test_model(t_model, False, args.workers)
#     print(f"As O: total={stats_o['total']}, wins={stats_o['wins']}, draws={stats_o['draws']}, losses={stats_o['losses']}")
#     print()
#     # save logs
    # with open('model_endings.log', 'a') as f:
    #     f.write(f'# {model_name}\n')
    #     f.write('# Model played as X\n')
    #     for x_moves, o_moves, is_x, status in logs_x:
    #         f.write(f"X_moves={x_moves}, O_moves={o_moves}, model_is_x={is_x}, status={status}\n")
    #     f.write('# Model played as O\n')
    #     for x_moves, o_moves, is_x, status in logs_o:
    #         f.write(f"X_moves={x_moves}, O_moves={o_moves}, model_is_x={is_x}, status={status}\n")
    # print('Testing completed. Logs written to model_endings.log')


# if __name__ == '__main__':
#     main()
