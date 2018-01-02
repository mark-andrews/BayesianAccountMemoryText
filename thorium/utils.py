#=============================================================================
# Numerical, data-analytic, plotting imports
#=============================================================================
import numpy
import pandas
import matplotlib
import datetime

# Some seemingly unimportant but annoying warnings are being raised by 
# seaborn==0.6.0, possibly when used with matplotlib==1.5.0
# We'll suppress them.
import warnings
warnings.filterwarnings('ignore')

import seaborn

#=============================================================================
# Rpy2 imports
#=============================================================================
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

pandas2ri.activate()
R = ro.r
lme = importr('lme4')

#================================ End Imports ================================

def ilogit(w):

    """
    Inverse logit.
    """

    return 1/(1 + numpy.exp(-w))


def logit(p):

    """
    Logit, or log odds, function.

    """

    return numpy.log(p) - numpy.log(1-p)


def rnorm(N=1, mean=0.0, sd=1.0):

    """
    N draws from normal distribution mean=mean, sd=sd.
    """

    return numpy.random.normal(size=N, loc=mean, scale=sd)

def runif(lower_bound=0, upper_bound=1.0, sample_size=1):
    
    """
    Draw `sample_size` samples from uniform distribution between
    lower_bound and upper_bound.

    """
    
    return numpy.random.uniform(lower_bound, 
                                upper_bound, 
                                size=sample_size)

def rbern(p):

    return 1 * (runif(sample_size=len(p)) < p)


def multilevel_logistic_regression(formula, data):

    return lme.glmer(formula, family='binomial', data=data)


def aictab(M):
    
    """
    Return the AICtab from a lme4 glmer as a dict.
    """
    
    aictab_robject = R.summary(M).rx2('AICtab')
    return dict(zip(list(aictab_robject.names), list(aictab_robject)))


def approx_log_bayes_factor(M_0, M_1):

    """

    BIC approximation of log of the Bayes factor
    
                   / P(D|M_0) \
     BF_{01} = log|  -------- |
                   \ P(D|M_1) /
    
    where P(D|M_0) is marginal likelihood of D under model M_0 and similar
    definition for P(D|M_1) and assuming equal prior probabilities for M_0 and
    M_1.
    
    From Section 4.1.3, page 778, Kass & Raftery (1995) "Bayes Factors".
    Journal of American Statistical Association. Vol 90, Number 430, pp
    773-795.
    
    BF_{01} =approx= 
    
        log(P(D|theta_0, M_0)) 
        - log(P(D|theta_1, M_1)) 
        - 1/2 * (k_0 - k_1) * log(n)
    
    where log(P(D|theta_0, M_0)) is log likelihood of model M_0 given theta_0, 
    the maximum likelihood estimate of parameters, and with a similar definition 
    for log(P(D|theta_1, M_1)). And k_0 and k_1 are the number of parameters of
    M_0 and M_1 and n is the number of observations. 


    BIC_0 = -2 * log(P(D|theta_0, M_0)) + k_0 * log(n)
    BIC_1 = -2 * log(P(D|theta_1, M_1)) + k_1 * log(n)
    
    Therefore, -(BIC_0 - BIC_1)/2 =approx= BF_{01}.
    
    """

    bic_0 = aictab(M_0)['BIC']
    bic_1 = aictab(M_1)['BIC']

    return -(bic_0 - bic_1)/2



class RandomFunctions(object):

    def rnorm(self, N=1, mean=0.0, sd=1.0):
    
        """
        N draws from normal distribution mean=mean, sd=sd.
        """

        return self.random.normal(size=N, loc=mean, scale=sd)


    def runif(self, lower_bound=0, upper_bound=1.0, sample_size=1):
        
        """
        Draw `sample_size` samples from uniform distribution between
        lower_bound and upper_bound.

        """
        
        return self.random.uniform(lower_bound, 
                                   upper_bound, 
                                   size=sample_size)
        
    def rbern(self, p):
        
        return 1 * (self.runif(sample_size=len(p)) < p)

    def choice(self, x):

        return self.random.choice(x)


class AucTest(RandomFunctions):
    
    def __init__(self, N=1000, a=0.0, b=1.0, random_seed=None):
        
        self.N = N
        self.a = a
        self.b = b
        
        self.random = numpy.random.RandomState(random_seed)

    def _auc(self, samples=10000):

        self.x = self.rnorm(self.N)
        self.p = ilogit(self.a + self.b*self.x)
        self.y = self.rbern(self.p) 
            
        where_y_one = numpy.where(self.y==1)[0]
        where_y_zero = numpy.where(self.y==0)[0]
        
        results = []
        for _ in xrange(samples):
            
            x_zero = self.x[self.choice(where_y_zero)]
            x_one = self.x[self.choice(where_y_one)]
            
            results.append(x_zero < x_one )
            
        return numpy.mean(results)
    
    @classmethod
    def auc(cls, 
            iterations=100, 
            a_choices=(0.0, 1.0),
            b_choices=(0.0, 0.1, 0.5, 1.00),
            N=10**3, 
            samples=10**4, 
            random_seed=None):

        seed_generator = SeedGenerator(random_seed)

        cls_random_seed = seed_generator.randint()

        random = numpy.random.RandomState(cls_random_seed)

        results = []
        for iteration in xrange(iterations):
            
            a = random.choice(a_choices)
            b = random.choice(b_choices)

            iteration_random_seed = seed_generator.randint()

            auc_test = cls(N=N, 
                           a=a, 
                           b=b, 
                           random_seed=iteration_random_seed)
            
            auc = auc_test._auc(samples)
            results.append([a, b, auc, iteration_random_seed])

        return pandas.DataFrame(results, columns=['a', 'b', 'auc', 'seed'])


    @classmethod
    def aucplot(cls, df):

        seaborn.lmplot(data=df,
                       x="b",
                       y="auc",
                       lowess=True,
                       size=5,
                       aspect=2) 


class ParameterSampler(object):
    
    """
    A class for randomly generating parameters for generating random data
    for testing the statistical power of a multilevel logistic 
    regression model.
    
    """
    
    def __init__(self,
                 J_choices = (25, 50, 75, 100, 125, 150),
                 K_choices = (5, 10, 20, 30, 40, 50),
                 K_min_choices = (3, 5),
                 N_choices = (10, 20),
                 subject_variation_choices = (0.1, 0.25, 0.5),
                 text_variation_choices = (0.1, 0.25),
                 effect_choices = (0.25, 0.5, 0.75, 1.0)):
        
        self.J_choices = J_choices
        self.K_choices = K_choices
        self.K_min_choices = K_min_choices
        self.N_choices = N_choices
        self.subject_variation_choices = subject_variation_choices
        self.text_variation_choices = text_variation_choices
        self.effect_choices = effect_choices

        self.random = numpy.random.RandomState()
    
    def sample_parameter_set(self, random_parameter_seed=None):
        
        """
        
        Generate a single set of random parameters.
        
        A single parameter set is a dictionary with the following keys:

        J: Integer giving number of subjects.

        K: Integer giving the total number of texts.

        K_min: Integer giving the minimum number of texts per subject.

        N: Integer giving number of items per text.

        subject_variation: 2-tuple of non-negative floats giving the 
                           variance of inter-subject variability.
                           The 2-tuple is (random intercept, random slope) variance.

        text_variation: 2-tuple non-negative floats giving 
                        variance of inter-text variability.
                        The 2-tuple is (random intercept, random slope) variance. 

        effect: 2-tuple giving the intercept 
               and slope coefficients of the model.

        iterations: Integer for how many data sets 
                    to generate from this set of parameters.
                   
        """

        self.random.seed(random_parameter_seed)
        choice = self.random.choice
        
        parameter_set = dict(J = choice(self.J_choices),
                             K = choice(self.K_choices),
                             N = choice(self.N_choices),
                             subject_variation = (0.5, choice(self.subject_variation_choices)),
                             text_variation = (0.5, choice(self.text_variation_choices)),
                             effect = (0.0, choice(self.effect_choices)))
        
        parameter_set['K_min'] = min(parameter_set['K'], 
                                     choice(self.K_min_choices))

        return parameter_set
    
    def sample(self, random_parameter_seeds):
        
        """
        Generate a list of random parameter sets; one for each parameter seed.
        """

        return [self.sample_parameter_set(random_parameter_seed)
                for random_parameter_seed in random_parameter_seeds]


class TestDataSampler(RandomFunctions):

    @classmethod
    def generate_test_data(cls,
                           J = 120, 
                           K = 10, 
                           N = 20, 
                           subject_variation=(0.1, 0.1),
                           text_variation=(0.1, 0.1),
                           effect=(0.0, 1.0),
                           K_min=3,
                           random_data_seed=None):

        test_data_sampler = cls(J=J,
                                K=K,
                                N=N,
                                subject_variation=subject_variation,
                                text_variation=text_variation,
                                effect=effect,
                                K_min=K_min)

        return test_data_sampler.sample_test_data(random_data_seed)

    def __init__(self, 
                 J = 120, 
                 K = 10, 
                 N = 20, 
                 subject_variation=(0.1, 0.1),
                 text_variation=(0.1, 0.1),
                 effect=(0.0, 1.0),
                 K_min=3):

        assert K_min <= K

        self.J = J
        self.K = K
        self.N = N
        self.subject_variation = subject_variation
        self.text_variation = text_variation
        self.effect = effect
        self.K_min = K_min

        self.random = numpy.random.RandomState()


    def sample_test_data(self, random_data_seed):

        self.random.seed(random_data_seed)

        alpha_null, beta_null = self.effect

        alpha_subject = self.rnorm(self.J, sd=self.subject_variation[0])
        alpha_text = self.rnorm(self.K, sd=self.text_variation[0])

        beta_subject = self.rnorm(self.J, sd=self.subject_variation[1])
        beta_text = self.rnorm(self.K, sd=self.text_variation[1])

        ys = []
        xs = []
        texts = []
        subjects = []
        words = []
        ws = []

        x = {}
        w = {}

        for k in xrange(self.K):
            x[k] = self.rnorm(self.N)

        for j in xrange(self.J):
            for k in self.random.permutation(self.K)[:self.K_min]:

                a = alpha_null + alpha_subject[j] + alpha_text[k]
                b = beta_null + beta_subject[j] + beta_text[k]

                w[k] = a + b*x[k] 

                p = ilogit(w[k])
                y = self.rbern(p)

                ws.extend(w[k])
                ys.extend(y)
                xs.extend(x[k])
                texts.extend([k]*self.N)
                subjects.extend([j]*self.N)
                words.extend([str(k) + '-' + str(n) for n in xrange(self.N)])

        data = dict(y=ys,
                    x=xs,
                    w=ws,
                    text=texts,
                    subject=subjects,
                    words=words)

        data = pandas.DataFrame(data, 
                                columns=['y', 'x', 'w', 'subject', 'text', 'words'])

        return data


class SeedGenerator(object):

    """
    Generate sets of integers. Never the same integer more than once.

    """
    
    maxint = numpy.iinfo(numpy.uint32).max
    
    def __init__(self, seed):
        
        self.seed = seed
        self.initialize()
        
    def initialize(self):
        
        self._randints = {}
        self.random = numpy.random.RandomState(self.seed)
    
    def randint(self):
        
        while True:
            _randint = self.random.randint(self.maxint)
            if _randint not in self._randints:
                break
                
        self._randints[_randint] = None
        
        return _randint
    
    def reset(self):
        self.initialize()

    def randints(self, N):
        
        return [self.randint() for _ in xrange(N)]
    
    def tuple_of_randints(self, N):
        
        return ([self.randint() for _ in xrange(N)],
                [self.randint() for _ in xrange(N)])


def make_task_batches(simulation_settings, random_seed):

    """
    Make a set of task batches on the basis of the information in the
    simulation_settings dict.

    Create the random seeds using the seed of seeds `random_seed`.

    """

    seed_generator = SeedGenerator(random_seed)

    task_batch = {}
    for key, settings in simulation_settings.items():

        number_of_trials, sim_parameters = settings

        random_parameter_seeds, random_data_seeds\
            = seed_generator.tuple_of_randints(number_of_trials)

        parameter_sampler = ParameterSampler(**sim_parameters)

        task_batch[key] = zip(random_data_seeds, 
                              parameter_sampler.sample(random_parameter_seeds))


    return task_batch

def power_test(arg_tuple):

    """

    Given the parameter dictionary `params`, generate a test data set using
    test_data_set and calculate the BIC for two multilevel logistic regression
    model. Both models have random slope and random intercepts that vary with
    subjects and by texts. One model has a single predictor variable `x`, while
    the other, the null model, does not.

    """

    random_data_seed, params = arg_tuple

    data = TestDataSampler.generate_test_data(random_data_seed=random_data_seed, 
                                              **params)

    random_intercepts = '(1|subject) + (1|text) + (1|words)'
    random_slopes = '(0+x|subject) + (0+x|text)'
    random_effects = '+'.join([random_intercepts, random_slopes])

    M = multilevel_logistic_regression('y ~ x +' + random_effects, data)
    M_null = multilevel_logistic_regression('y ~ 1 +' + random_effects, data)

    bf = approx_log_bayes_factor(M, M_null)

    result = [params['J'],
              params['K'],
              params['K_min'],
              params['N'],
              params['effect'][0],
              params['effect'][1],
              params['subject_variation'][0],
              params['subject_variation'][1],
              params['text_variation'][0],
              params['text_variation'][1],
              random_data_seed,
              bf]

    return result


def simulation_results_to_dataframe(simulation_results):
    
    """
    Convert simulation results to data frame.
    """

    X = numpy.array(simulation_results)
    column_labels = ['J', 
                     'K', 
                     'K_min', 
                     'N', 
                     'a', 
                     'b', 
                     'subj a', 
                     'subj b', 
                     'text a', 
                     'text b',
                     'data seed',
                     'bf']
    
    return pandas.DataFrame(X, columns=column_labels)


def powerplot(df, evidence_strength='very strong'):

    """
    Produce a seaborn facet plot.

    """

    # Interpretations of 2 x log BF 
    evidence_strength_thresholds = {'strong': 6.0,
                                    'very strong': 10.0}

    bayes_factor_threshold = evidence_strength_thresholds[evidence_strength]

    aggfunc=lambda x: numpy.mean(2*x > bayes_factor_threshold) 

    df_pivot = pandas.pivot_table(df, 
                                  index=['J' , 'K', 'b', 'subj b'], 
                                  values=['bf'], 
                                  aggfunc=aggfunc).reset_index()

    
    seaborn.set_context("notebook", 
                        font_scale=1.0, 
                        rc={"lines.linewidth": 1.5})
    seaborn.set_style("darkgrid")
    seaborn.set(font_scale=1.5)
    
    grid = seaborn.FacetGrid(df_pivot, 
                             col="J", 
                             hue="subj b", 
                             row='K', 
                             size=5, 
                             legend_out= True)

    grid.map(matplotlib.pyplot.axhline, y=0.8, linewidth=1, ls='dotted', color='k')
    grid.map(matplotlib.pyplot.plot, "b", "bf", marker="o", ms=5)
    grid.set(xticks=[0.0, 0.25, 0.5, 0.75, 1.0], 
             yticks=[0.0, 0.25, 0.5, 0.75, 1.0],
             xlim=(-0.1, 1.1), 
             ylim=(-0.1, 1.1))

    grid.fig.tight_layout(w_pad=1)
    grid.add_legend(title='Subject variability')
    
    return grid

def get_save_filename(key, random_seed=None):

    """
    Make a filename for saving results from simulations.

    """
    if not (random_seed is None):
        random_seed_str = 'seed_' + str(random_seed)
    else:
        random_seed_str = ''

    timestamp = datetime.datetime.today().strftime('%b%d-%H%M')
    return key + '-' + random_seed_str + '-' + timestamp + '.csv'
