python scripts/run_eval.py --dataset gsm --answer_type float --max_gens 40 --temperature 0.7 --prompt_type code --prompt_file math_prompts
python scripts/run_eval.py --dataset asdiv --answer_type float --max_gens 40 --temperature 0.7 --prompt_type code --prompt_file math_prompts
python scripts/run_eval.py --dataset svamp --answer_type float --max_gens 40 --temperature 0.7 --prompt_type code --prompt_file math_prompts
python scripts/run_eval.py --dataset date --answer_type str --max_gens 40 --temperature 0.7 --prompt_type code --prompt_file date_understanding_prompt
python scripts/run_eval.py --dataset tracking_three --answer_type str --max_gens 40 --temperature 0.7 --prompt_type text --prompt_file tracking_three --end "# Q:"
python scripts/run_eval.py --dataset ld_three --answer_type str --max_gens 40 --temperature 0.7 --prompt_type text --prompt_file ld_three
python scripts/run_eval.py --dataset strategy_qa --answer_type str --max_gens 40 --temperature 0.7 --prompt_type text --prompt_file strategy_qa_prompt
python scripts/run_eval.py --dataset boolean_expressions --answer_type str --max_gens 40 --temperature 0.7 --prompt_type text --prompt_file boolean_expressions
python scripts/run_eval.py --dataset snarks --answer_type str --max_gens 40 --temperature 0.7 --prompt_type text --prompt_file snarks
python scripts/run_eval.py --dataset ruin_names --answer_type str --max_gens 40 --temperature 0.7 --prompt_type text --prompt_file ruin_names
python scripts/run_eval.py --dataset salient_translation --answer_type str --max_gens 40 --temperature 0.7 --prompt_type text --prompt_file salient_translation
python scripts/run_eval.py --dataset disambiguation_qa --answer_type str --max_gens 40 --temperature 0.7 --prompt_type text --prompt_file disambiguation_qa
python scripts/run_eval.py --dataset penguins_in_a_table_text --answer_type str --max_gens 40 --temperature 0.7 --prompt_type text --prompt_file penguins_in_a_table