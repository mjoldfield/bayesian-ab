import math
import random
import json

from scipy import stats

from bernoulli import BernoulliArm

from bayesab   import BayesAB
from ucb1      import UCB1
from annealing import AnnealingEpsilonGreedy

def std_tests():
  algos = [   [ 'ucb1',    lambda: UCB1([],[]) ]
              , [ 'aeg',     lambda: AnnealingEpsilonGreedy([],[]) ]
              , [ 'bayes',   lambda: BayesAB(0)   ]
  ]

  for (n_runs, n_steps) in [ (1000001,100), (100001,1000), (10001,10000), (1001,100000) ]:
    for p in [0.01, 0.03, 0.09, 0.11, 0.3, 0.9]:
        run_tests('std', [0.1, p], n_runs, n_steps, algos)

def cau_tests():
  algos = [  [ 'bayes0',   lambda: BayesAB(0)    ]
          ,  [ 'bayesm1',  lambda: BayesAB(-1.0) ]
          ,  [ 'bayesp1',  lambda: BayesAB( 1.0) ]
  ]

  for (n_runs, n_steps) in [ (100001,100), (10001,1000), (1001,10000), (101,100000) ]:
    for p in [0.01, 0.03, 0.09, 0.11, 0.3, 0.9]:
        run_tests('cau', [0.1, p], n_runs, n_steps, algos)
        
def sm_tests():
  algos = [  [ 'bayes',    lambda: BayesAB(0, False)  ]
          ,  [ 'bayessm',  lambda: BayesAB(0, True)   ]
  ]

  for (n_runs, n_steps) in [ (100001,100), (10001,1000), (1001,10000), (101,100000) ]:
    for p in [0.01, 0.03, 0.09, 0.11, 0.3, 0.9]:
        run_tests('sm', [0.1, p], n_runs, n_steps, algos)

def bozo_tests():
  algos = [  [ 'norm',    lambda: BayesAB(0, False)  ]
          ,  [ 'smth',    lambda: BayesAB(0, True,  False)   ]
          ,  [ 'bozo',    lambda: BayesAB(0, True,  True)    ]
  ]

  for (n_runs, n_steps) in [ (100001,100), (10001,1000), (1001,10000), (101,100000) ]:
    for p in [0.01, 0.03, 0.09, 0.11, 0.3, 0.9]:
        run_tests('bozo', [0.1, p], n_runs, n_steps, algos)

        
def run_tests(tag, means, n_runs, n_steps, algos):

    print("tag: %s, means: %s, n_steps: %d, n_runs %d"
          % (tag, str(means), n_steps, n_runs))

    arms = list(map(lambda mu: BernoulliArm(mu), means))
    
    report = {}
    
    for [name, mk_algo] in algos:
        ft_scores = []
        ht_ratios = []

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
            ft = results[n     -1]['score']
            ht = results[int(n/2)]['score']
            
            ft_scores.append(ft)
            if ft > 0:
                ht_ratios.append(ht / ft)

        ft_scores.sort()

        report[name] = mk_report(name, ft_scores, ht_ratios, n_tries)

    filename = ("res/%s-%0.3f-%0.3f-%06d-%06d.json"
                % (tag, means[0], means[1], n_steps, n_runs))
    with open(filename, 'w') as fp:
        json.dump(report, fp)
        print("wrote " + filename)
        
    print("--\n")
    
def mk_report(name, ft_scores, ht_ratios, n_tries):
    i_med = int((len(ft_scores) - 1) / 2)
    i_lq  = i_med - int(i_med / 2)
    i_uq  = i_med + int(i_med / 2)

    mean_ft = sum(ft_scores) / len(ft_scores)
    ht_ft   = sum(ht_ratios) / len(ht_ratios)
    
    f_arm0  = float(n_tries[0]) / float(n_tries[0] + n_tries[1])
    
    print("%-12s  %6.3f   %6.3f  %8d %8d %8d  %12.3f"
          % (name, f_arm0, ht_ft, ft_scores[i_lq], ft_scores[i_med], ft_scores[i_uq], mean_ft))
    
    return { 'f_arm0': f_arm0
             , 'ft_score_lq': ft_scores[i_lq]
             , 'ft_score_med': ft_scores[i_med]
             , 'ft_score_uq': ft_scores[i_uq]
             , 'ft_score_mean': mean_ft
    }


    

def run_test(algo, arms, n_steps):

    n_arms = len(arms)
    algo.initialize(n_arms)
    
    results = []

    score   = 0
    for t in range(n_steps):
        i_arm = algo.select_arm()

        d = arms[i_arm].draw()
        algo.update(i_arm, d)

        score += d
        results.append({ 'arm': i_arm, 'draw': d, 'score': score })

    return results

#sm_tests()
#std_tests()
#cau_tests()
bozo_tests()

