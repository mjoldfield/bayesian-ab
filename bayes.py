import math
import random
from functools import lru_cache

# log of n factorial
def log_n_fac(n):
  # for n < 2048 do this directly (cached); for larger ones switch to
  # Stirling's formula

  # log_n_fac(2047) =          
  #    Mathematica: 13564.3263533846767473821855165 
  #    direct:      13564.326353384677
  #    Stirling:    13564.326353383847

  if n < 2048:
    return log_n_fac_direct(n)
  else:
    return 0.5 * math.log(2 * math.pi * n) + n * (math.log(n) - 1.0) + math.log(1 + 1 / (12 * n))

@lru_cache(maxsize=None)
def log_n_fac_direct(n):
    return math.log(math.factorial(n))

def log_binomial(n,k):
  return log_n_fac(n) - (log_n_fac(k) + log_n_fac(n - k))

# compare a & b then return
#   x_lt if a  < b
#   x_eq if a == b
#   x_gt if a  > b

def ternary_cmp(a, b, x_lt, x_eq, x_gt):
  if   (a < b):
    return x_lt
  elif (a > b):
    return x_gt
  else:
    return x_eq

class BayesVirtual():
  def __init__(self, explore_on_tie=False):
    self.explore_on_tie = explore_on_tie
    return

  def initialize(self, n_arms):
    if (n_arms != 2):
      raise ValueError("Can only handle n_arms == 2")
    self.stats = [ { 'n': 0, 'k': 0 } for i in range(n_arms) ]
        
  def update(self, i_arm, score):
    self.stats[i_arm]['n'] += 1
    if score:
      self.stats[i_arm]['k'] += 1

  def select_arm(self):
    [a1,a2] = self.stats

    na = a1['n']
    ka = a1['k']

    nb = a2['n']
    kb = a2['k']

    log_ev1 = (log_binomial(na, ka) 
               + log_binomial(nb,kb) 
               - log_binomial(na + nb, ka + kb) 
	             - math.log(na + nb + 1) 
              ) 

    log_ev2 = -(math.log(1 + na) + math.log(1 + nb))

    if self.explore_on_tie:
      explore = log_ev1 >= log_ev2
    else:
      explore = log_ev1 >  log_ev2

    # store the relative probability of H2 for diagnostics
    
    # log_ratio +ve => h2 more probable than h1
    #               => good to exploit and not explore

    # log_ratio unlikely to go very -ve (H1 only beats
    # H2 by Occam factor), so this should be well behaved:
    log_ratio = log_ev2 - log_ev1
    self.pr_h2 = 1.0 / (1.0 + math.exp(-log_ratio))

    if explore:
        return self.select_arm_explore(na, ka, nb, kb)
    else:
        return self.select_arm_exploit(na, ka, nb, kb)

  def diag(self):
    return { 'pr(h2)': self.pr_h2 }

  def select_arm_explore(self, na, ka, nb, kb):
    raise Exception('BayesVirt is a pure virtual class: do not invoke it')

  def select_arm_exploit(self, na, ha, nb, hb):
    return ternary_cmp(ha * nb, hb * na, 1, random.randint(0,1), 0)

class Bayes(BayesVirtual):
  def select_arm_explore(self, na, ha, nb, hb):
    return ternary_cmp(na, nb, 0, random.randint(0,1), 1)

class BayesRnd(BayesVirtual):

  def select_arm_explore(self, na, ha, nb, hb):
    return random.randint(0,1)    
    
class BayesExtExpl(BayesVirtual):

  def select_arm_explore(self, na, ha, nb, hb):
    lhs = (ha + 1) * (nb + 2) 
    rhs = (hb + 1) * (na + 2) 
                                  
    h = ha + hb		      
    n = na + nb		      

    (u,v) = ternary_cmp(2 * h, n, (lhs,rhs), (0,0), (rhs,lhs))

    return ternary_cmp(u,v,1,random.randint(0,1),0)
