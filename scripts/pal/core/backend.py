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

import openai
import time
import os

openai.api_key = os.getenv('OPENAI_API_KEY')
openai.organization = os.getenv('OPENAI_API_ORG')



# GPT-3 API
def call_gpt(prompt, model='code-davinci-002', stop=None, temperature=0., top_p=1.0,
        max_tokens=128, majority_at=None, logprobs = 0):
    num_completions = majority_at if majority_at is not None else 1
    num_completions_batch_size = 5
    
    completions = []
    all_data = []
    for i in range(20 * (num_completions // num_completions_batch_size + 1)):
        try:
            requested_completions = min(num_completions_batch_size, num_completions - len(completions))
            if model == "gpt-3.5-turbo":
                # from pdb import set_trace; set_trace()
                ans = openai.ChatCompletion.create(
                    model = model,
                    max_tokens = max_tokens,
                    # stop = stop,
                    messages = [{"role": "user", "content": prompt},],
                    temperature = temperature,
                    top_p = top_p,
                    n = requested_completions,
                    # best_of = requested_completions
                )
                # from pdb import set_trace; set_trace()

                completions.extend([choice['message']['content'] for choice in ans['choices']])
            else:
                ans = openai.Completion.create(
                                model=model,
                                max_tokens=max_tokens,
                                stop=stop,
                                prompt=prompt,
                                temperature=temperature,
                                top_p=top_p,
                                n=requested_completions,
                                logprobs = logprobs,
                                best_of=requested_completions)
                # from pdb import set_trace as bp
                # bp()
                all_data.extend([choice['logprobs'] for choice in ans['choices']])
                completions.extend([choice['text'] for choice in ans['choices']])
            if len(completions) >= num_completions:
                if logprobs !=0:
                    return completions[:num_completions], all_data[:num_completions]
                else:
                    return completions[:num_completions]
        except openai.error.RateLimitError as e:
            print(e, type(e))
            print('Sleeping', min(i**2, 60))
            time.sleep(min(i**2, 60))
        except openai.error.InvalidRequestError as e:
            print(e, type(e))
            max_tokens = int(max_tokens // 2)
            continue
        except Exception as e:
            print(e, type(e))
            # 3/0
            print('Sleeping', min(i**2, 60))
            time.sleep(min(i**2, 60))
            continue
    raise RuntimeError('Failed to call GPT API')
