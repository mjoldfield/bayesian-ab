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

def select_on(a,b):
  if   b > a:
    return 1
  elif b < a:
    return 0
  else:
    return random.randint(0,1)


class BayesAB():
  def __init__(self, exp_method='', marginalize=False, expl_on_tie=False):
    self.exp_method    = exp_method
    self.marginalize   = marginalize
    self.expl_on_tie   = expl_on_tie
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

    log_ratio = self.evidence_ratio(n1,k1, n2,k2)

    # log_ratio unlikely to go very -ve (H1 only beats
    # H2 by Occam factor), so this should be well behaved:
    self.pr_h2 = 1.0 / (1.0 + math.exp(-log_ratio))

    # log_ratio +ve => h2 more probable than h1
    #               => good to exploit and not explore

    if self.marginalize:
      explore = random.random() < self.pr_h2
    elif self.expl_on_tie:
      explore = log_ratio <= 0.0
    else:
      explore = log_ratio < 0.0

    if explore:
      if   self.exp_method == 'random':
        return random.randint(0,1)
      elif self.exp_method == 'min_n':
        return select_on(n2, n1)
      elif self.exp_method == 'max_tr':
        if (2 * (k1 + k2) >= (n1 + n2)):
          q1,q2 = k1,k2
        else:
          q1,q2 = n1 - k1, n2 - k2

        return 1 - select_on(q1 * n2, q2 * n1)

      elif self.exp_method == 'max_tr1':
        if (2 * (k1 + k2) >= (n1 + n2)):
          q1,q2 = k1,k2
        else:
          q1,q2 = n1 - k1, n2 - k2

        return 1 - select_on((q1 + 1) * (n2 + 2), (q2 + 1) * (n1 + 2))
      
      elif self.exp_method == 'min_ph2':
        # algebra shows this is the same as min_tr
        
        # delta is most likely outcome of the toss
        if (2 * (k1 + k2) >= (n1 + n2)):
          delta = 1
        else:
          delta = 0

        # best(ish) guess at evidence ratio after the next toss
        ratio1 = self.evidence_ratio(n1 + 1, k1 + delta, n2, k2)
        ratio2 = self.evidence_ratio(n1, k1, n2 + 1, k2 + delta)

        # choose arm which makes H1 more likely!
        return select_on(ratio2, ratio1)

      else:
        raise Exception('Invalid exploration type')

    else:
      return select_on(k1 * n2, k2 * n1)

  def evidence_ratio(self, n1, k1, n2, k2):

    logph1 = (log_binomial(n1, k1)
              + log_binomial(n2,k2)
              - log_binomial(n1 + n2, k1 + k2)
              - math.log(n1 + n2 + 1))
    
    logph2 = -(math.log(1 + n1) + math.log(1 + n2))

    # log(pr(H2) / pr(H1))
    log_ratio = logph2 - logph1

    return log_ratio

  def diag(self):
    return { 'pr(h2)': self.pr_h2 }
