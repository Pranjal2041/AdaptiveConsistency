import numpy as np
from typing import List, Dict
from collections import Counter
from scipy import integrate, stats


class StoppingCriterias:

    def __init__(self, *args, **kwargs):

        ...

    def should_stop(self, *args, **kwargs) -> Dict:
        ...


class BetaStoppingCriteria(StoppingCriterias):

    def __init__(self, conf_thresh : float = 0.95) -> None:
        super().__init__()
        self.conf_thresh = conf_thresh

    def should_stop(self, answers : List, conf_thresh : int = None, verbose : bool = False) -> Dict:
        
        if conf_thresh is None: conf_thresh = self.conf_thresh

        
        most_common = Counter(answers).most_common(2)
        if len(most_common) == 1:
            a, b = most_common[0][1], 0
        else:
            a, b= most_common[0][1], most_common[1][1]
        a = float(a)
        b = float(b)

        return_dict = {
            'most_common' : most_common[0][0],
            'prob' : -1,
            'stop' : False,
        }
            

        try:
            prob =  integrate.quad(lambda x : x**(a) * (1-x)**(b), 0.5, 1)[0] / integrate.quad(lambda x : x**(a) * (1-x)**(b), 0, 1)[0]
        except Exception as e:
            # print error message
            print(f"Error during numerical integration: {e}")
            return_dict['stop'] = False
            return_dict['prob'] = -1
            return return_dict
        return_dict['prob'] = prob
        return_dict['stop'] = prob >= conf_thresh
        return return_dict

class RandomStoppingCriteria(StoppingCriterias):

    def __init__(self, conf_thresh : float = 0.1) -> None:
        super().__init__()
        self.conf_thresh = conf_thresh

    def should_stop(self, answers : List, conf_thresh : int = None, verbose : bool = False) -> Dict:
        
        if conf_thresh is None: conf_thresh = self.conf_thresh

        return_dict = {
            'most_common' : Counter(answers).most_common(1)[0][0],
            'prob' : 0,
            'stop' : np.random.uniform(0,1) < conf_thresh,
        }
        return return_dict
    
class EntropyStoppingCriteria(StoppingCriterias):

    def __init__(self, conf_thresh : float = 0.75) -> None:
        super().__init__()
        self.conf_thresh = conf_thresh

    def should_stop(self, answers : List, conf_thresh : int = None, verbose : bool = False) -> Dict:
        
        if conf_thresh is None: conf_thresh = self.conf_thresh

        counter = dict(Counter(answers))
        lis = list(counter.values())
        if len(lis) < 2:
            lis.append(1)
        entropy = stats.entropy(lis, base = 2)
        return_dict = {
            'most_common' : Counter(answers).most_common(1)[0][0],
            'prob' : -1,
            'stop' : False,
        }
        if len(answers) != 1:
            return_dict['stop'] = entropy/np.log2(len(lis)) <= conf_thresh
            return_dict['prob'] = entropy/np.log2(len(lis))
    
        return return_dict
        
class MajorityStoppingCriteria(StoppingCriterias):

    def __init__(self, conf_thresh : float = 0.8) -> None:
        super().__init__()
        self.conf_thresh = conf_thresh

    def should_stop(self, answers : List, conf_thresh : int = None, verbose : bool = False) -> Dict:
        
        if conf_thresh is None: conf_thresh = self.conf_thresh

        return_dict = {
            'most_common' : Counter(answers).most_common(1)[0][0],
            'prob' : -1,
            'stop' : False,
        }
        if len(answers) != 1:
            return_dict['stop'] = Counter(answers).most_common(1)[0][1]/len(answers) >= conf_thresh
            return_dict['prob'] = Counter(answers).most_common(1)[0][1]/len(answers)
    
        return return_dict
    
class DirichletStoppingCriteria(StoppingCriterias):

    def __init__(self, conf_thresh : float = 0.95, top_k_elements : int = 5, use_markov : bool = True) -> None:
        super().__init__()
        self.conf_thresh = conf_thresh
        self.top_k_elements = top_k_elements
        self.use_markov = use_markov

    def integrate_mcs(self, f, limits, N = 10000):
        ranges = []
        samples = []

        for _, funcs in enumerate(limits[::-1]):

            if len(samples) == 0:
                ranges.append(funcs())
            else:
                ranges.append(funcs(*samples[::-1]))
                # TODO: Note, we assume, that the first value is actually a scalar.
                try:
                    ranges[-1][0] = ranges[-1][0][0]
                except: ...

            samples.append(np.random.uniform(*ranges[-1], size=N))
        integrand_values = f(*samples) * np.prod([r[1] - r[0] for r in ranges], axis=0)

        integral_approximation = (1/N) * np.sum(integrand_values)
        return integral_approximation


    def should_stop(self, answers : List, conf_thresh : int = None, verbose : bool = False) -> Dict:
        
        if conf_thresh is None: conf_thresh = self.conf_thresh

        counts = dict(Counter(answers))
        if len(counts) < 3:
            return BetaStoppingCriteria(conf_thresh).should_stop(answers, conf_thresh, verbose)
        
        most_common = Counter(answers).most_common(2)[0][0]
        counts = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=False)[-self.top_k_elements:]}
        len_counts = len(counts)

        functions = []
        functions2  =[]
        for i, _ in enumerate(counts.items()):
            if i == len_counts - 2:
                break
            if self.use_markov:
                functions.append(lambda *args: [np.array([0 for _ in range(args[0].shape[0])]), np.max([np.array([0 for _ in range(args[0].shape[0])]), np.min([np.array([0.5 for _ in range(args[0].shape[0])]), 1 - np.sum(args, axis = 0) - np.max(args, axis = 0), (1-np.sum(args, axis = 0))/2], axis = 0)], axis = 0)])
            else:
                functions.append(lambda *args: [0, max(0, min(0.5, 1 - sum(args) - max(args), (1-sum(args))/2))])
            functions2.append(lambda *args: [0, max(0, min(0.5, 1 - sum(args) - max(args), (1-sum(args))/2))])

        # Outermost limit
        functions.append(lambda *args: [0, 0.5])
        functions2.append(lambda *args: [0, 0.5])

        denom_functions = []
        for i, _ in enumerate(counts.items()):
            if i == len_counts - 2:
                break
            denom_functions.append(lambda *args: [0, 1-np.sum(args, axis = 0)])
        # Outermost limit
        denom_functions.append(lambda *args: [0, 1])
        exec(
            f'''def integrand({",".join(["a" + str(i) for i in range(len(functions))])}):
                counts = {counts}
                ks = list(counts.keys())
                args = [{",".join(["a" + str(i) for i in range(len(functions))])}]

                outp = np.prod([args[i] ** counts[k] for i, k in enumerate(list(counts.keys())[:-1])], axis = 0) * (1 - np.sum(args, axis = 0)) ** counts[list(counts.keys())[-1]]
                return outp
            '''
        )

        return_dict = {
            'most_common' : most_common,
            'prob' : -1,
            'stop' : False,
        }

        try:
            # print('Computing Integration')
            opts = {}
            opts = {'limit': 3, 'epsrel' : 1e-1,'epsabs': 1e-1}

            if self.use_markov:
                N = min(500000, 5000 * 2**len(functions))
                N = 50000 * 1 if len(functions) <= 4 else 50000 * (2** ((len(functions) - 3)//2))
                prob = self.integrate_mcs(locals()['integrand'], functions, N = N) / self.integrate_mcs(locals()['integrand'], denom_functions, N = N)
            else:
                prob = integrate.nquad(locals()['integrand'], functions, opts = opts)[0] / integrate.nquad(locals()['integrand'], denom_functions, opts = opts)[0]
            return_dict['prob'] = prob
            return_dict['stop'] = prob >= conf_thresh

        except Exception as e:
            # print error message
            print(f"Error during numerical integration: {e}")
        
        return return_dict
    
class AlwaysFalseStoppingCriteria(StoppingCriterias):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def should_stop(self, answers : List, *args, **kwargs) -> Dict:
        return {
            'most_common' : Counter(answers).most_common(1)[0][0],
            'prob' : -1,
            'stop' : False,
        }
    