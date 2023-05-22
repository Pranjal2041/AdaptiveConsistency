import importlib

prompt_file = 'math_prompts'
math_prompts = importlib.import_module(f'pal.prompt.{prompt_file}')
print(math_prompts)