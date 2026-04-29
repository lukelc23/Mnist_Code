import torch
import pandas as pd
from collections import defaultdict

def evaluate_by_pair(model, device, dataset):
    model.eval()

    correct_by_config = defaultdict(int)
    total_by_config = defaultdict(int)

    item_to_position = {item: pos for pos, item in enumerate(dataset.ordering)}

    with torch.no_grad():
        for i in range(len(dataset)):
            stimulus, label = dataset[i]

            pair_idx = i // 2
            pair = dataset.pairs[pair_idx % len(dataset.pairs)]
            winner, loser = pair[0], pair[1]

            is_flipped = i % 2
            if is_flipped:
                first_item, second_item = loser, winner
            else:
                first_item, second_item = winner, loser

            stimulus = stimulus.unsqueeze(0).to(device)
            output = model(stimulus)
            pred = output.argmax(dim=1).item()

            key = (winner, loser, first_item, second_item)
            total_by_config[key] += 1
            if pred == label:
                correct_by_config[key] += 1

    rows = []
    for key in sorted(total_by_config.keys()):
        winner, loser, first_item, second_item = key
        acc = correct_by_config[key] / total_by_config[key]
        w_pos = item_to_position[winner]
        l_pos = item_to_position[loser]
        rows.append({
            'winner_item': winner,
            'loser_item': loser,
            'winner_position': w_pos,
            'loser_position': l_pos,
            'first_item': first_item,
            'second_item': second_item,
            'first_position': item_to_position[first_item],
            'second_position': item_to_position[second_item],
            'distance': l_pos - w_pos,
            'accuracy': acc,
            'correct': correct_by_config[key],
            'total': total_by_config[key]
        })

    df = pd.DataFrame(rows)

    total_correct = sum(correct_by_config.values())
    total = sum(total_by_config.values())
    print(f"Overall: {total_correct}/{total} ({100 * total_correct / total:.1f}%)")

    return df