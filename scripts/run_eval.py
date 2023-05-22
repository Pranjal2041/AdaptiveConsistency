import copy
import json
import argparse
import tqdm
import os

import sys

from pal import interface, runtime
# from pal.prompt import math_prompts



parser = argparse.ArgumentParser()
parser.add_argument('--append', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--dataset', default='gsm', type=str)
parser.add_argument('--model', default='code-davinci-002', type=str)
parser.add_argument('--temperature', default=0.0, type=float)
parser.add_argument('--top_p', default=1.0, type=float)
parser.add_argument('--max_tokens', default=256, type=int)
parser.add_argument('--prompt_file', default="math_prompts", type=str)
parser.add_argument('--end', default="\n\n\n", type=str)
parser.add_argument('--prompt_type', default='code', type=str)
parser.add_argument('--vicuna_url', default=None, type=str)
parser.add_argument('--start_data', default=None, type=int)
parser.add_argument('--end_data', default=None, type=int)
parser.add_argument('--conf_thresh', default=0.99, type = float)
parser.add_argument('--max_gens', default=40, type = int)
parser.add_argument('--seed', default=1, type = int)
parser.add_argument('--answer_type', default='float', type = str, help='Type of answer to expect. One of float or str')
parser.add_argument('--stop_criteria', default=None, type = str, help='AdaptiveConsistency stop criteria to use. Defaults to Self-Consistency')
parser.add_argument('--stop_criteria_thresh', default=0.95, type = float, help='AdaptiveConsistency stop criteria threshold to use. See AdaptiveConsistency for details')


args = parser.parse_args()

import importlib
math_prompts = importlib.import_module(f'pal.prompt.{args.prompt_file}')


DATA_PATH = f'datasets/{args.dataset}.jsonl'
if not os.path.exists(DATA_PATH):
    DATA_PATH = f'datasets/{args.dataset}.json'
if DATA_PATH.endswith('.jsonl'):
    examples = list(map(json.loads, open(DATA_PATH)))
elif DATA_PATH.endswith('.json'):
    examples = json.load(open(DATA_PATH))['examples']

dataset_name = args.dataset
if args.start_data is not None and args.end_data is None:
    examples = examples[args.start_data:]
    dataset_name += f'_{args.start_data}_end'
elif args.start_data is None and args.end_data is not None:
    examples = examples[:args.end_data]
    dataset_name += f'_0_{args.end_data}'
elif args.start_data is not None and args.end_data is not None:
    examples = examples[args.start_data:args.end_data]
    dataset_name += f'_{args.start_data}_{args.end_data}'

OUTPUT_PATH = f'outputs/{args.model}/{dataset_name}/{dataset_name}_{args.max_gens}_{args.temperature}_stop{"self" if args.stop_criteria is None else args.stop_criteria}_seed{args.seed}.jsonl'
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)



if args.prompt_type != 'code' and args.prompt_type != 'text':
    print(f'Unknown prompt type: {args.prompt_type}')
    print('Defaulting to code prompt')
    args.prompt_type = 'code'

answer_type = args.answer_type
# answer_type = 'str' if args.dataset.find('date')!=-1 else 'float'
if args.prompt_type == 'code':

    # PAL style prompting
    if args.dataset.find('date')!=-1:
        itf = interface.AdaptiveProgramInterface(
            step_size = 1,
            max_gens=args.max_gens,
            runtime = runtime.DateRuntime(),
            stop=args.end,
            model=args.model,
            verbose=args.verbose,
            openai_url=args.vicuna_url,
            answer_type=answer_type,
            stop_criteria = args.stop_criteria,
            stop_criteria_thresh = args.stop_criteria_thresh,
        )
    else:
        itf = interface.AdaptiveProgramInterface(
            step_size = 1,
            max_gens=args.max_gens,
            stop=args.end,
            get_answer_expr='solution()',
            model=args.model,
            verbose=args.verbose,
            openai_url=args.vicuna_url,
            answer_type=answer_type,
            stop_criteria = args.stop_criteria,
            stop_criteria_thresh = args.stop_criteria_thresh,
        )


elif args.prompt_type == 'text':
    # CoT style prompting
    itf = interface.AdaptiveTextInterface(
        step_size = 1,
        max_gens=args.max_gens,
        stop=args.end,
        model=args.model,
        openai_url=args.vicuna_url,
        stop_criteria = args.stop_criteria,
        stop_criteria_thresh = args.stop_criteria_thresh,
    )
        


if args.append:
    lines = open(OUTPUT_PATH).readlines()
    num_skip_exps = len(lines)
    scores = [x['score'] for x in map(json.loads, lines)]
else:
    num_skip_exps = 0
    scores = []

with open(OUTPUT_PATH, 'a' if args.append else 'w') as f:
    pbar = tqdm.tqdm(examples[num_skip_exps:], initial=num_skip_exps, total=len(examples))
    for x in pbar:
        question = x['input']
        result = copy.copy(x)
        
        try:
            ans, answers = itf.run(math_prompts.MATH_PROMPT.format(question=question),
                temperature=args.temperature, top_p=args.top_p,
                max_tokens=args.max_tokens)
            if answer_type == 'float':
                ans = float(ans)
                score = 1 if abs(ans - x['target']) < 1e-3 else 0
            else:
                score = 1 if ans == x['target'] else 0
        except Exception as e:
            print('Error',e)
            ans = ''
            # Failed to load any answers
            answers = []
            score = 0
        scores.append(score)
        
        result['answer'] = ans
        result['score'] = score
        result['generation'] = itf.history
        result['answers'] = answers
        f.write(json.dumps(result) + '\n')
        
        itf.clear_history()
        f.flush()

print(f'Accuracy - {sum(scores) / len(scores)}')
