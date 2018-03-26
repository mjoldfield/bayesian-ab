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
              , [ 'bayes',   lambda: BayesAB(False)      ]
              , [ 'bla',     lambda: BayesAB(True)       ]
              , [ 'ble',     lambda: BayesAB(True, True) ]
  ]

  for (n_runs, n_steps) in [ (101,100), (101,1000) ]: # [ (1000001,100), (100001,1000) ]: #, (10001,10000), (1001,100000) ]:
    for p in [0.01, 0.03, 0.09, 0.11, 0.3, 0.9]:
        run_tests('std', [0.1, p], n_runs, n_steps, algos)
        run_tests('std', [p, 0.1], n_runs, n_steps, algos)
        
def run_tests(tag, means, n_runs, n_steps, algos):

    print("tag: %s, means: %s, n_steps: %d, n_runs %d"
          % (tag, str(means), n_steps, n_runs))

    arms = list(map(lambda mu: BernoulliArm(mu), means))
    
    report = {}
    
    for [name, mk_algo] in algos:
        scores = []

        n_tries = [0,0]
        
        for i in range(n_runs):
            algo  = mk_algo()

            results = run_test(algo, arms, n_steps)

            do_log  = i < 100
            if do_log:
                logname = ("log/%s-%0.3f-%0.3f-%06d-%06d-%s-%02d.json"
                            % (tag, means[0], means[1], n_steps, n_runs, name, i))
                with open(logname, 'w') as fp:
                    json.dump(results, fp)

            for r in results:
                n_tries[r['arm']] += 1
            
            n = len(results)
            score = results[n-1]['score']
            
            scores.append(score)

        scores.sort()

        report[name] = mk_report(name, scores, n_tries)

    filename = ("res/%s-%0.3f-%0.3f-%06d-%06d.json"
                % (tag, means[0], means[1], n_steps, n_runs))
    with open(filename, 'w') as fp:
        json.dump(report, fp)
        print("wrote " + filename)
        
    print("--\n")
    
def mk_report(name, scores, n_tries):
    i_med = int((len(scores) - 1) / 2)
    i_lq  = i_med - int(i_med / 2)
    i_uq  = i_med + int(i_med / 2)

    score_mean = sum(scores) / len(scores)

    f_arm0  = float(n_tries[1]) / float(n_tries[0] + n_tries[1])
    
    print("%-12s  %6.3f   %8d %8d %8d  %12.3f"
          % (name, f_arm0, scores[i_lq], scores[i_med], scores[i_uq], score_mean))
    
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

std_tests()
