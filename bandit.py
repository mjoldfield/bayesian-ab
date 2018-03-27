import math
import random
import json

from scipy import stats

from bernoulli import BernoulliArm

from bayesab   import BayesAB
from ucb1      import UCB1
from annealing import AnnealingEpsilonGreedy

def std_tests():
  algos = [   [ 'ucb1',      lambda: UCB1([],[]) ]
              , [ 'aeg',     lambda: AnnealingEpsilonGreedy([],[]) ]
              , [ 'bayes',   lambda: BayesAB('min_n')   ]
#              , [ 'ble',     lambda: BayesAB('min_ph2') ]
              , [ 'blr',     lambda: BayesAB('max_tr')  ]
              , [ 'bls',     lambda: BayesAB('max_tr1') ]
              , [ 'rnd',     lambda: BayesAB('random')  ]
  ]

  for (n_runs, n_steps) in [ (100000,100), (100000, 1000) ]:
    for p in [0.01, 0.03, 0.09, 0.11, 0.3, 0.9]:
        run_tests('std', [0.1, p], n_runs, n_steps, algos)
        
def run_tests(tag, means, n_runs, n_steps, algos):

    print("tag: %s, means: %s, n_steps: %d, n_runs 2 * %d = %d"
          % (tag, str(means), n_steps, n_runs, 2 * n_runs))

    arms = list(map(lambda mu: BernoulliArm(mu), means))
    
    report = {}
    
    for [name, mk_algo] in algos:
        scores = []
        score_by_swap = {'r': 0, 'n': 0}

        n_tries = [0,0]

        for swap in [False,True]:

          for i in range(n_runs):

            algo  = mk_algo()

            if swap:
              arms_used = list(reversed(arms))
              arms_tag  = "r"
            else:
              arms_used = arms
              arms_tag  = "n"
              
            results = run_test(algo, arms_used, n_steps)

            if swap:
              for r in results:
                r['arm'] = 1 - r['arm']

            do_log  = i < 100
            if do_log:
                logname = ("log/%s-%0.3f-%0.3f-%06d-%06d-%s-%s-%02d.json"
                            % (tag, means[0], means[1], n_steps, n_runs, name, arms_tag, i))
                with open(logname, 'w') as fp:
                    json.dump(results, fp)

            for r in results:
                n_tries[r['arm']] += 1
            
            n = len(results)
            score = results[n-1]['score']
            score_by_swap[arms_tag] += score
            
            scores.append(score)

        scores.sort()

        report[name] = mk_report(name, scores, score_by_swap, n_tries)

    filename = ("res/%s-%0.3f-%0.3f-%06d-%06d.json"
                % (tag, means[0], means[1], n_steps, n_runs))
    with open(filename, 'w') as fp:
        json.dump(report, fp)
        print("wrote " + filename)
        
    print("--\n")
    
def mk_report(name, scores, score_by_swap, n_tries):
    n = len(scores)

    i_med = int((n - 1) / 2)
    i_lq  = i_med - int(i_med / 2)
    i_uq  = i_med + int(i_med / 2)

    score_mean = sum(scores) / n

    f_arm0  = float(n_tries[1]) / float(n_tries[0] + n_tries[1])

    norm_score = score_by_swap['n'] / (0.5 * n)
    swap_score = score_by_swap['r'] / (0.5 * n)
    
    print("%-12s  %12.3f %12.3f     %6.3f     %8d %8d %8d    %12.3f"
          % (name, norm_score, swap_score, f_arm0,
             scores[i_lq], scores[i_med], scores[i_uq], score_mean))
    
    return { 'f_arm0':       f_arm0
             , 'score_lq':   scores[i_lq]
             , 'score_med':  scores[i_med]
             , 'score_uq':   scores[i_uq]
             , 'score_mean': score_mean
    }

    
def run_test(algo, arms, n_steps):

    n_arms = len(arms)
    algo.initialize(n_arms)
    
    results = []

    has_diags = callable(getattr(algo, 'diag', None))

    score   = 0
    for t in range(n_steps):
        i_arm = algo.select_arm()

        d = arms[i_arm].draw()
        algo.update(i_arm, d)

        score += d
        h = { 'arm': i_arm, 'draw': d, 'score': score }

        if has_diags:
          h.update(algo.diag())

        results.append(h)

    return results

if __name__ == "__main__":
  std_tests()
