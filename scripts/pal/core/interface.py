# Copyright 2022 PAL Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import signal
from contextlib import redirect_stdout
from typing import Any, Callable, List, Optional
from collections import Counter

from .runtime import GenericRuntime
from .backend import call_gpt
from .vicuna import call_vicuna

from adaptive_consistency import AC, stop_criteria_dict



def init_adaptive_consistency(max_gens, stop_criteria, stop_criteria_thresh):
    if stop_criteria is None:
        stop_criteria = 'always_false'
    if stop_criteria_thresh is None or stop_criteria_thresh == -1:
        ac = AC(max_gens = max_gens, stop_criteria=stop_criteria_dict[stop_criteria]())
    else:
        ac = AC(max_gens = max_gens, stop_criteria=stop_criteria_dict[stop_criteria](conf_thresh = stop_criteria_thresh))
    return ac


class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def timeout_handler(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)


class TextInterface:
    
    def __init__(
        self,
        max_gens: int = 40,
        model: str = 'code-davinci-002',
        answer_prefix: str = 'answer is',
        stop: str = '\n\n\n',
        extract_answer: Optional[Callable[[str], Any]] = None,
        openai_url: Optional[str] = None,
        stop_criteria: Optional[str] = None,
        stop_criteria_thresh: Optional[float] = None,
    ):
        self.max_gens = max_gens
        self.ac = init_adaptive_consistency(self.max_gens, stop_criteria, stop_criteria_thresh)

        self.history = []
        self.answer_prefix = answer_prefix
        self.extract_answer_fn = extract_answer
        self.stop = stop
        self.model = model
        self.openai_url = openai_url
        # Hacky solution:
        if self.openai_url is not None:
            globals()['call_gpt'] = lambda *args, **kwargs : call_vicuna(*args, **kwargs, url=openai_url)
        
    def reinit(self):
        ...


    def clear_history(self):
        self.history = []
    
    def extract_answer(self, gen: str):
        if self.extract_answer_fn:
            return self.extract_answer_fn (gen)
        last_line = gen.strip().split('\n')[-1]
        # TODO: Searching for last line is not at all necessary!
        last_idx = last_line.rfind(self.answer_prefix)
        if last_idx == -1:
            return ""
        answer = last_line[last_idx + len(self.answer_prefix):].strip()
        if answer.endswith('.'):
            answer = answer[:-1]
        return answer
    
    def execute(self, gen: str):
        if isinstance(gen, List):
            gen = '\n'.join(gen)
        return self.extract_answer(gen)

    def run(self, prompt, temperature=0.0, top_p=1.0, majority_at=None, max_tokens=512, logprobs=0):
        # gen = call_gpt(prompt, model=self.model, stop=self.stop, 
            # temperature=temperature, top_p=top_p, max_tokens=max_tokens, majority_at=majority_at)
        if logprobs != 0:
            gens, dt = call_gpt(prompt, model=self.model, stop=self.stop, 
                    temperature=temperature, top_p=top_p, max_tokens=max_tokens, majority_at=majority_at, logprobs=logprobs)   
        else:
            gens = call_gpt(prompt, model=self.model, stop=self.stop, 
                temperature=temperature, top_p=top_p, max_tokens=max_tokens, majority_at=majority_at, )
            
        if logprobs != 0:
            self.history.append([gens, dt])
        else:
            self.history.append(gens)
        results = []
        for gen in gens:
            results.append(self.extract_answer(gen))
        return Counter(results).most_common(1)[0][0]
        

class ProgramInterface:
    
    def __init__(
        self,
        max_gens: int = 40,
        model: str = 'code-davinci-002',
        runtime: Optional[Any] = None,
        stop: str = '\n\n',
        get_answer_symbol: Optional[str] = None,
        get_answer_expr: Optional[str] = None,
        get_answer_from_stdout: bool = False,
        verbose: bool = False,
        openai_url: Optional[str] = None,
        stop_criteria: Optional[str] = None,
        stop_criteria_thresh: Optional[float] = None,
    ) -> None:

        self.max_gens = max_gens
        self.ac = init_adaptive_consistency(self.max_gens, stop_criteria, stop_criteria_thresh)

        self.model = model
        self.runtime = runtime if runtime else GenericRuntime()
        self.history = []
        self.stop = stop
        self.answer_symbol = get_answer_symbol
        self.answer_expr = get_answer_expr
        self.get_answer_from_stdout = get_answer_from_stdout
        self.verbose = verbose

        if openai_url is not None:
            globals()['call_gpt'] = lambda *args, **kwargs : call_vicuna(*args, **kwargs, url=openai_url)

    def reinit(self):
        import copy
        self.runtime._global_vars = copy.copy(self.runtime.GLOBAL_DICT) 

    
    def clear_history(self):
        self.history = []
    
    def process_generation_to_code(self, gens: str):
        return [g.split('\n') for g in gens]
    
    def generate(self, prompt: str, temperature: float =0.0, top_p: float =1.0, 
            max_tokens: int =512, majority_at: int = None, logprobs = 0):
        if logprobs != 0:
            gens, dt = call_gpt(prompt, model=self.model, stop=self.stop, 
                temperature=temperature, top_p=top_p, max_tokens=max_tokens, majority_at=majority_at, logprobs=logprobs)   
        else:
            gens = call_gpt(prompt, model=self.model, stop=self.stop, 
                temperature=temperature, top_p=top_p, max_tokens=max_tokens, majority_at=majority_at, )
        if self.verbose:
            print(gens)
        code = self.process_generation_to_code([x.strip() for x in gens])
        if logprobs != 0:
            self.history.append([gens, dt])
        else:
            self.history.append(gens)
        return code
    
    def execute(self, code: Optional[List[str]] = None, TIMEOUT = 2):
        # from pdb import set_trace
        # set_trace()
        with timeout(TIMEOUT):
            code = code if code else self.code
            if self.get_answer_from_stdout:
                program_io = io.StringIO()
                with redirect_stdout(program_io):
                    self.runtime.exec_code('\n'.join(code))
                program_io.seek(0)
                return program_io.readlines()[-1]
            elif self.answer_symbol:
                self.runtime.exec_code('\n'.join(code))
                return self.runtime._global_vars[self.answer_symbol]
            elif self.answer_expr:
                self.runtime.exec_code('\n'.join(code))
                return self.runtime.eval_code(self.answer_expr)
            else:
                self.runtime.exec_code('\n'.join(code[:-1]))
                return self.runtime.eval_code(code[-1])
        return ""
    
    def run(self, prompt: str, time_out: float =10, temperature: float =0.0, top_p: float =1.0, 
            max_tokens: int =512, majority_at: int = None, prepend_to_code = "", logprobs = 0):
        code_snippets = self.generate(prompt, majority_at=majority_at, temperature=temperature, top_p=top_p, max_tokens=max_tokens, logprobs = logprobs)
        # print(code_snippets)
        results = []
        for code in code_snippets:
            self.reinit()
            with timeout(time_out):
                try:
                    exec_result = self.execute(prepend_to_code.splitlines() + code)
                except Exception as e:
                    print(e)
                    continue
                results.append(exec_result)
        counter = Counter(results)
        return counter.most_common(1)[0][0]



class AdaptiveProgramInterface(ProgramInterface):

    def __init__(self, answer_type = 'float', step_size = 1,  *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.answer_type = answer_type
        self.step_size = step_size

    def generate(self, prompt: str, temperature: float =0.0, top_p: float =1.0, 
            max_tokens: int =512, majority_at: int =None, logprobs = 0):
        self.history.append([])
        gens = call_gpt(prompt, model=self.model, stop=self.stop, 
            temperature=temperature, top_p=top_p, max_tokens=max_tokens, majority_at=majority_at, )
        if self.verbose:
            print(gens)
        gens = [x.strip() for x in gens]
        # print('Processing generations to code')
        code = self.process_generation_to_code(gens)
        # print('Appending to code')
        self.history[-1].extend(gens)
        return code

    def run(self, prompt: str, time_out: float =10, temperature: float =0.0, top_p: float =1.0, 
            max_tokens: int =512, majority_at: int =None, prepend_to_code = ""):
        all_results = []
        for i in range(0, self.max_gens, self.step_size):
            code_snippets = self.generate(prompt, majority_at=self.step_size, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            
            results = []
            for code in code_snippets:
                self.reinit()
                with timeout(time_out):
                    try:
                        exec_result = self.execute(prepend_to_code.splitlines() + code)
                        if self.answer_type == 'float':
                            exec_result = float(exec_result)
                        else:
                            exec_result = str(exec_result)
                    except Exception as e:
                        print('Eror', e)
                        # traceback.print_exc()

                        continue
                    results.append(exec_result)
            all_results += results
            # print(all_results)
            if len(all_results) == 0:
                continue
            # if has_conclusive_majority_binomial_prob(all_results, self.conf_thresh)[1]:
            if self.ac.should_stop(all_results):
                # print('Less goo!', results)
                break
        print('Used {} generations'.format(i+4))
        counter = Counter(all_results)
        most_common = counter.most_common(1)[0]
        return most_common[0], all_results
    

class AdaptiveTextInterface(TextInterface):
    def __init__(self, step_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = []
        self.step_size = step_size

    
    def run(self, prompt: str, time_out: float =10, temperature: float =0.0, top_p: float =1.0, 
            max_tokens: int =512, majority_at: int =None, prepend_to_code = ""):
        all_results = []
        for i in range(0, self.max_gens, self.step_size):
            print(i)
            gens = call_gpt(prompt, model=self.model, stop=self.stop, 
                    temperature=temperature, top_p=top_p, max_tokens=max_tokens, majority_at=self.step_size, )
            print(i)
            results = []
            for gen in gens:
                self.reinit()
                self.history.append(gen)
                ans = self.extract_answer(gen)
                results.append(ans)
            all_results += results
            if len(all_results) == 0:
                continue
            # if has_conclusive_majority_binomial_prob(all_results, self.conf_thresh)[1]:
            if self.ac.should_stop(all_results):
                break
        print('Used {} generations'.format(i+4))
        counter = Counter(all_results)
        most_common = counter.most_common(1)[0]
        return most_common[0], all_results