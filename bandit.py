import math
import random
import json

from scipy import stats

from bernoulli import BernoulliArm

#from bayesab   import BayesAB
from ucb1      import UCB1
from annealing import AnnealingEpsilonGreedy

import bayes

def argmax(xs):
  return max(enumerate(xs), key=lambda x:x[1])[0]

def std_tests():
  algos = [   [ 'ucb1',      lambda: UCB1([],[]) ]
              , [ 'aeg',     lambda: AnnealingEpsilonGreedy([],[]) ]
#              , [ 'bayes',   lambda: bayes.Bayes() ]
#              , [ 'rnd',     lambda: bayes.BayesRnd() ]
#              , [ 'bls',     lambda: bayes.BayesExtExpl() ]
              , [ 'bayesT',  lambda: bayes.Bayes(True) ]
              , [ 'rndT',    lambda: bayes.BayesRnd(True) ]
              , [ 'blsT',    lambda: bayes.BayesExtExpl(True) ]
              , [ 'bayesF',  lambda: bayes.Bayes(False) ]
              , [ 'rndF',    lambda: bayes.BayesRnd(False) ]
              , [ 'blsF',    lambda: bayes.BayesExtExpl(False) ]
#              , [ 'bayes',   lambda: BayesAB('min_n')   ]
#              , [ 'bls',     lambda: BayesAB('max_tr1') ]
#              , [ 'rnd',     lambda: BayesAB('random')  ]
#              , [ 'bayesE',   lambda: BayesAB('min_n', False, True)   ]
#              , [ 'blsE',     lambda: BayesAB('max_tr1', False, True) ]
#              , [ 'rndE',     lambda: BayesAB('random', False, True)  ]
  ]

  n_runs = 1000
  for n_warmup in [ 0, 1000 ]:
    for n_steps in [ 100, 1000, 10000 ]:
      for p in [0.01, 0.03, 0.09, 0.11, 0.3, 0.9]:
        run_tests('std', [0.1, p], n_warmup, n_runs, n_steps, algos)
        
def run_tests(tag, means, n_warmup, n_runs, n_steps, algos):

    print("tag: %s, warmup: %d, means: %s, n_steps: %d, n_runs 2 * %d = %d"
          % (tag, n_warmup, str(means), n_steps, n_runs, 2 * n_runs))

    arms = [ BernoulliArm(mu) for mu in means ]
    best_arm = argmax(means)
    
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
              warmup_arm = 1
            else:
              arms_used = arms
              arms_tag  = "n"
              warmup_arm = 0
              
            results = run_test(algo, arms_used, [(warmup_arm, n_warmup)] , n_steps)

            if swap:
              for r in results:
                r['arm'] = 1 - r['arm']

            do_log  = i < 100
            if do_log:
                logname = ("log/%s-%06d-%0.3f-%0.3f-%06d-%06d-%s-%s-%02d.json"
                            % (tag, n_warmup, means[0], means[1], n_steps, n_runs, name, arms_tag, i))
                with open(logname, 'w') as fp:
                    json.dump(results, fp)

            for r in results:
                n_tries[r['arm']] += 1
            
            n = len(results)
            score = results[n-1]['score']
            score_by_swap[arms_tag] += score
            
            scores.append(score)

        scores.sort()

        report[name] = mk_report(name, scores, score_by_swap, n_tries, best_arm)

    filename = ("res/%s-%06d-%0.3f-%0.3f-%06d-%06d.json"
                % (tag, n_warmup, means[0], means[1], n_steps, n_runs))
    with open(filename, 'w') as fp:
        json.dump(report, fp)
        print("wrote " + filename)
        
    print("--\n")
    
def mk_report(name, scores, score_by_swap, n_tries, best_arm):
    n = len(scores)

    i_med = int((n - 1) / 2)
    i_lq  = i_med - int(i_med / 2)
    i_uq  = i_med + int(i_med / 2)

    score_mean = sum(scores) / n

    tot_tries  = float(n_tries[0] + n_tries[1])

    f_bestarm  = float(n_tries[best_arm]) / tot_tries
    f_arm1     = float(n_tries[1])        / tot_tries

    norm_score = score_by_swap['n'] / (0.5 * n)
    swap_score = score_by_swap['r'] / (0.5 * n)
    
    print("%-12s %8.3f %8.3f    %6.3f   %6d %6d %6d   %8.3f   %6.3f"
          % (name, norm_score, swap_score, f_arm1,
             scores[i_lq], scores[i_med], scores[i_uq], score_mean, f_bestarm))
    
    return { 'f_arm1':       f_arm1
             , 'f_good_arm': f_bestarm
             , 'score_lq':   scores[i_lq]
             , 'score_med':  scores[i_med]
             , 'score_uq':   scores[i_uq]
             , 'score_mean': score_mean
    }

    
def run_test(algo, arms, warmups, n_steps):

    n_arms = len(arms)
    algo.initialize(n_arms)
    
    results = []

    has_diags = callable(getattr(algo, 'diag', None))

    for (arm,n) in warmups:
      for i in range(n):
        d = arms[arm].draw()
        algo.update(arm, d)

    
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
  random.seed(1770)
  std_tests()
