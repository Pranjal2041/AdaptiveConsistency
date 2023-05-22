from tqdm import tqdm
import argparse
from adaptive_consistency import AC, stop_criteria_dict
import json

def main(dt, ac, min_gens = 1, max_gens = 40, eval_as_str = False):
    
    correct_answers = 0
    total_answers = len(dt)
    total_gens = 0
    for _, x in tqdm(enumerate(dt), total = len(dt)):
        x = {k: (v[:max_gens] if isinstance(v, list) else v) for k, v in x.items() }
        for m in range(min_gens, len(x['scores'])+1):
            answers = x['answers'][:m]
            new_answers = []
            for _, xx in enumerate(answers):
                try: 
                    if eval_as_str:
                        if str(xx).strip()=='':
                            continue
                        new_answers.append(str(xx))
                    else:
                        new_answers.append(float(xx))
                except: ...
            if len(new_answers) == 0: 
                if m == len(x['scores']):
                    total_gens += m
                continue
            outp = ac.should_stop(new_answers, return_dict = True)
            majority_val, majority_bool = outp['most_common'], outp['stop']


            if majority_bool or (m == len(x['scores'])):
                total_gens += m
                try:
                    if eval_as_str:
                        if str(majority_val).strip() == str(x['target']).strip():
                            correct_answers += 1
                            break
                    else:
                        if abs(float(str(majority_val).strip()) - float(x['target'])) < 1e-3:
                            correct_answers += 1
                            break
                except Exception as e:
                    print('Error', majority_val, m, e)
                    break
                break
    return correct_answers, total_answers, total_gens

if __name__ == '__main__':

    # Usage: python examples/eval_outputs.py --output_file examples/outputs/outputs.jsonl --stop_criteria beta --stop_criteria_thresh 0.95

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--stop_criteria', type=str, default=None)
    parser.add_argument('--stop_criteria_thresh', type=float, required=False, default=None)

    args = parser.parse_args()

    if args.stop_criteria is None:
        args.stop_criteria = 'always_false'
        print('No Stop Criteria Provided. Running Self-Consistency')
    if args.stop_criteria_thresh is None or args.stop_criteria_thresh == -1:
        ac = AC(max_gens = 1000, stop_criteria=stop_criteria_dict[args.stop_criteria]())
    else:
        ac = AC(max_gens = 1000, stop_criteria=stop_criteria_dict[args.stop_criteria](conf_thresh = args.stop_criteria_thresh))
    

    dt = list(map(json.loads, open(args.output_file)))

    eval_as_str = not ('gsm' in args.output_file or 'asdiv' in args.output_file or 'svamp' in args.output_file)

    correct_answers, total_answers, total_gens = main(dt, ac, eval_as_str = eval_as_str)
    print(f'Accuracy: {correct_answers}/{total_answers} ({correct_answers/total_answers*100:.2f}%)')
    print(f'Average Gens: {total_gens/total_answers:.2f}')