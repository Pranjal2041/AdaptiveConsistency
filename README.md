# Adaptive Consistency: 
A Library for Running efficient LLM generation using [Adaptive-Consistency](https://github.com/Pranjal2041/AdaptiveConsistency.git) in your code.

## Introduction
Introduction Here

## Installation

### From PyPi

```bash
pip install AdaptiveConsistency
```

### From Source

First clone the repo:
```bash
git clone https://github.com/Pranjal2041/AdaptiveConsistency.git
```

Next install the package using: 
```bash 
python setup.py install
```

## Usage

Using Adaptive Consistency in your code requires only 2-3 lines of changes in your existing framework.

### 1. Importing the library

```python
from AdaptiveConsistency import AC, BetaStoppingCriteria
```

### 2. Initializing the library

```python
ac = AC(model, stopping_criteria=BetaStoppingCriteria(0.95), max_gens = 40)
```

### 3. Using the library

You can directly run a whole loop of evaluation using:

```python
ac.eval_loop(sampling_function, *args, **kwargs)
```

(TODO: Add example of how to use openai, and rephrase also.)

Or you can check for consistency of answers at each step:

```python
answers = []
for i in range(40):
    answers.append(generate_answer_from_model())
    if ac.check_consistency(answers):
        break
```




## Citation

Details Coming Soon!

## License

TODO: Add License
