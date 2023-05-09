import numpy as np

from typing import List, Any

from .stopping_criterias import *

class AC:
    '''
    A class for using Adaptive Consistency for your LLM generations.

    Args:
        max_gens (int): Maximum number of generations to perform for each question.
        stop_criteria : StoppingCriterias: The stopping criteria function to use. 
        verbose (bool): Whether to print verbose output.

    Attributes:
        max_gens (int): Maximum number of generations to perform.
        verbose (bool): Whether to print verbose output.
        stop_criteria: The stopping criteria function to use.
    '''

    def __init__(self, max_gens : int = 40, stop_criteria = BetaStoppingCriteria, verbose : bool = False) -> None:
        '''
        Initializes an instance of the AC class.

        Args:
            max_gens (int): Maximum number of generations to perform.
            stop_criteria (StoppingCriterias): The stopping criteria function to use. 
            verbose (bool): Whether to print verbose output.
        '''

        self.max_gens = max_gens
        self.verbose = verbose
        self.set_stop_criteria(stop_criteria)


    def set_max_gens(self, max_gens : int) -> None:
        '''
        Sets the maximum number of generations per question.

        Args:
            max_gens (int): Maximum number of generations to perform.
        '''
        self.max_gens = max_gens

    def set_stop_criteria(self, stop_criteria : BetaStoppingCriteria) -> None:
        '''
        Sets the stopping criteria function.

        Args:
            stop_criteria (StoppingCriterias): The stopping criteria function to use. 
        '''
        if isinstance(stop_criteria, str):
            if stop_criteria == 'beta':
                self.stop_criteria = BetaStoppingCriteria()
            elif stop_criteria == 'dirichlet':
                self.stop_criteria = DirichletStoppingCriteria()
            elif stop_criteria == 'random':
                self.stop_criteria = RandomStoppingCriteria()
            elif stop_criteria == 'majority':
                self.stop_criteria = MajorityStoppingCriteria()
            elif stop_criteria == 'entropy':
                self.stop_criteria = EntropyStoppingCriteria()
            else:
                raise ValueError(f"Unknown stopping criteria: {stop_criteria}")

        elif isinstance(stop_criteria, StoppingCriterias):
            # The function is already initialized, so we can use it directly
            self.stop_criteria = stop_criteria
        elif isinstance(stop_criteria, type):
            # The function is not initialized, so we need to initialize it
            self.stop_criteria = stop_criteria()

    def is_consistent(self, answers : List[Any], return_dict : bool = False) -> bool:
        '''
        Checks if the answers are consistent based on Adaptive Consistency Algorithm and corresponding Stopping Criteria.

        Args:
            answers (List): A list of answers to check consistency.
            return_dict (bool): Whether to return the full dictionary of output.

        Returns:
            Union[bool, Dict]: Whether the answers are consistent or not. If return_dict is True, returns the full dictionary of output.
        ''' 

        if len(answers) > self.max_gens:
            # Raise a warning
            if self.verbose:
                warnings.warn(f"Warning: max_gens ({self.max_gens}) reached.")


        should_stop = self.stop_criteria.should_stop(answers, verbose=self.verbose)
        if return_dict:
            return should_stop
        else:
            return should_stop['stop']

    def eval_loop(self, eval_function, *args, **kwargs):
        '''
        Runs AdaptiveConsistency Algorithm by repeatedly calling the evaluation function until the stopping criteria is met.

        Args:
            eval_function: The function to evaluate.
            *args: Additional positional arguments to pass to the eval_function.
            **kwargs: Additional keyword arguments to pass to the eval_function.

        Returns:
            List: A list of answers generated from evaluation function using AdaptiveConsistency.
        '''
        answers = []
        for _ in range(self.max_gens):
            answer = eval_function(*args, **kwargs)
            answers.append(answer)
            if self.is_consistent(answers):
                return answers
            

