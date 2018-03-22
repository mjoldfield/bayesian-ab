import math
import random

# log of n factorial
def log_n_fac(n):
  # for small n do it directly; for larger ones switch to
  # Stirling's formula
  # log(30!) = 74.6582, approx = 74.6555
  if n < 30:
    return math.log(math.factorial(n))
  else:
    return 0.5 * math.log(2 * math.pi * n) + n * (math.log(n) - 1.0)

def log_binomial(n,k):
  return log_n_fac(n) - (log_n_fac(k) + log_n_fac(n - k))

class BayesAB():
  def __init__(self, caution, smooth=False):
    self.caution = caution
    self.smooth  = smooth
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
    n1 = a1['n']
    k1 = a1['k']

    n2 = a2['n']
    k2 = a2['k']
      
    logph1 = (log_binomial(n1, k1)
              + log_binomial(n2,k2)
              - log_binomial(n1 + n2, k1 + k2)
              - math.log(n1 + n2 + 1))
    
    logph2 = -(math.log(1 + n1) + math.log(1 + n2))

    # log(pr(H2) / pr(H1))
    # caution is a fiddle factor which makes H1 more probable
    # i.e. the algorithm becomes more cautious 
    log_ratio = logph2 - (logph1 + self.caution)

    if self.smooth:
      # choose whether to explore by randomly picking 
      # t is a small number to avoid numerical issues
      # - if either prob is less than t then don't sample
      t = 1e-6
      threshold = math.log((1-t) / t)
    else:
      # hard transition: just do what's more probable
      threshold = 0.0

    if   log_ratio > threshold:
      explore = False
    elif log_ratio < -threshold:
      explore = True
    else:
      explore = random.random() > (1.0 / (1.0 + math.exp(log_ratio)))

    if explore:
      # not sure which is best, so get more samples from rarer source
      if (n1 < n2):
        return 0
      elif (n1 > n2):
        return 1
    else:
      if   (k1 * n2 < k2 * n1):
        return 1
      elif (k1 * n2 > k2 * n1):
        return 0

    return random.randint(0,1)

