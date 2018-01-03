from __future__ import division, absolute_import

from itertools import cycle
import configobj
from numpy import zeros, array, arange, sqrt, dot, unique
from numpy import random
from gustav.samplers import fortransamplers
from numpy.random import randint, rand
import os

from . import utils


def get_experiment_texts(cfg_file, cache):

    '''Extract the `text_memoranda` from the cfg_file and return the texts as a
    dictionary.
    '''

    cfg_filepath = os.path.join(cache, cfg_file)

    texts = {}
    for key,value in configobj.ConfigObj(cfg_filepath)['text_memoranda'].items():
        texts[key] = value['text']

    return texts


def text_to_words(text, word_to_index):
    return [word for word in utils.tokenize(text) if word in word_to_index]


class PosteriorPredictive(object):

    '''

    A class for doing multiple posterior predictions for a given topic model,
    defined by the data and state.

    '''

    def __init__(self, data, state, verbose=False):

        self.verbose = verbose

        self.vocabulary = list(data['vocabulary'])
        self.V = len(self.vocabulary)

        self.word_to_index = {v:i for i, v in enumerate(self.vocabulary)}

        assert self.V == data['w'].max() + 1
        assert all(state['w'] == data['w'])

        self.K = state['K_rep']

        assert state['x'].max() == self.K - 1

        self.am = state['m'] * state['a']

        S = zeros((self.K, self.V), dtype=int)

        for xi, wi in zip(state['x'], state['w']):
            S[xi, wi] += 1

        phi = S + state['psi']*state['b']
        self.phi = (phi.T/phi.sum(1)).T

        self._set_seed()

    def _set_seed(self, seed=None):
        self.random = random.RandomState(seed=seed)

    def sample_vpi(self, text, iterations=1000, vpi_init=None, burn_in=False):

        '''
        Gibbs sampler to draw samples from posterior of vpi given `text`.

        If not `burn_in`, then collect all the samples and return. If
        `burn_in`, only return the last sample drawn from the chain.

        '''

        words = text_to_words(text, self.word_to_index)

        w = [self.word_to_index[word] for word in words
             if word in self.word_to_index]

        w = array(w)
        
        if vpi_init is None:
            vpi = self.random.dirichlet(self.am)
        else:
            vpi = vpi_init

        if not burn_in:
            Vpi = zeros((iterations, self.K))

        for iteration in xrange(iterations):

            Q = (self.phi[:, w].T * vpi).T
            Q = Q/Q.sum(0)

            _K, nj = Q.shape

            assert _K == self.K

            R = zeros(self.K)

            for xi in [sample(p=Q[:,i], choice=self.random.choice) for i in xrange(nj)]:
                R[xi] += 1
            
            vpi = self.random.dirichlet(R + self.am)

            if not burn_in:
                Vpi[iteration] = vpi

        if burn_in:
            return vpi
        else:
            return Vpi

    def posterior_prediction(self, 
                             text, 
                             seed=101,
                             burn_in_iterations=10000, 
                             iterations=25000, 
                             nchains=3,
                             max_attempts_to_converge=3,
                             rhat_max_threshold=1.01,
                             thin=1000):

        '''
        For a given text, return the posterior predictive distribution.

        Sample vpi from Posterior(vpi|text, data) as follows: 

        * Run `nchains` chains
        * Burn in for `burn_in` iterations
        * Then draw `iterations` samples
        * Check for convergence. Convergence is when gelman-rubin <= rhat_max_threshold
        * If not converged, draw another `iterations` samples and check
          convergence again. If not, repeat again.
        * When converged, downsample `thin` samples from the set of samples from
          all chains.
        
        Given the `thin` samples from vpi, then do, for each vpi_i
            w_i = \sum_{x} P(w|x, phi) * P(x|vpi_i)
        and then average as follows
            w = \sum_i w_i / thin

        '''

        self._set_seed(seed=seed)

        # burn in
        vpi_init = [None] * nchains
        for chain in xrange(nchains):
            vpi_init[chain] = self.sample_vpi(text, 
                                              iterations=burn_in_iterations, 
                                              vpi_init = None,
                                              burn_in=True)

        for _ in xrange(max_attempts_to_converge):

            vpi_list = []
            for chain in xrange(nchains):
                vpi_list.append(self.sample_vpi(text, 
                                                iterations=iterations,
                                                vpi_init=vpi_init[chain],
                                                burn_in=False))

            rhat_max = convergence_diagnostic(vpi_list).max()

            if rhat_max <= rhat_max_threshold:

                if self.verbose:
                    print('Converged')

                break

            else:

                if self.verbose:
                    print('rhat: %2.2f' % rhat_max)
                vpi_init = array([vpi[-1] for vpi in vpi_list])

        vpi = flatten(vpi_list)

        N, V = vpi.shape

        w = zeros(self.V)
        I = self.random.permutation(N)[:min(N,thin)]
        for p in vpi[I]:
            w += dot(p, self.phi)

        w = w/len(I)

        return w, rhat_max, _

class PosteriorPredictive2(PosteriorPredictive):

    '''

    A class for doing multiple posterior predictions for a given topic model,
    defined by the data and state.

    In this case, we use phi, and am estimates directly, rather than estimating
    them from the state.

    '''

    def __init__(self, data, phi, am, verbose=False):

        self.verbose = verbose

        self.vocabulary = list(data['vocabulary'])
        self.V = len(self.vocabulary)

        self.word_to_index = {v:i for i, v in enumerate(self.vocabulary)}

        assert self.V == data['w'].max() + 1
    
        self.K, _V = phi.shape
        assert self.V == _V
        assert self.K == len(am)
        
        self.am = am 
        self.phi = phi

        self._set_seed()

def topic2str(phi, vocabulary, K=25):
    return ','.join([vocabulary[k] for k in phi.argsort()[::-1][:K]])


def sample(p, choice, size=None):
    K = len(p)
    return choice(arange(K), p=p, size=size, replace=True)


def flatten(P):

    _P = []
    for p in P:
        _P.extend(p)

    return array(_P)


def convergence_diagnostic(vpi_list, high_mass_limit=0.99):

    '''
    Check convergence of vpi.
    In practice, vpi is a long vector, e.g. around 50000 elements. Its
    probability mass is usually concentrated on a small number of elements. 
    We don't care too much about the other elements so we just check the
    convergence of the high mass elements. We define the high mass elements as
    all those whose combined mass is `high_mass_limit`.

    '''

    n, K = vpi_list[0].shape

    var_j = array([p.var(0, ddof=1) for p in vpi_list])
    mean_j = array([p.mean(0) for p in vpi_list])

    W = var_j.mean(0)
    B = mean_j.var(0, ddof=1) * n

    var_alt = (n-1)/n * W + (1/n) * B

    _rhat =  sqrt(var_alt/W)

    _vpi = flatten(vpi_list)

    q = _vpi.mean(0)
    f = 0.0
    top_k = []
    for k in q.argsort()[::-1]:
        f += q[k]
        top_k.append(k)

        if f >= high_mass_limit:
            break

    return _rhat[array(top_k)]


class DirichletMultinomialCompound(object):

    """
    A Gibbs sampler for a Dirichlet multinomial compound model.

    """
    
    def __init__(self, S, inits=None):

        if inits is None:
            inits = {}
        
        self.S = S
        
        self.K, self.V = self.S.shape
        
        try:
            self.psi = inits['psi']
        except KeyError:
            _psi = rand(self.V)
            self.psi = _psi/_psi.sum()

        try:
            self.b = inits['b']
        except KeyError:
            self.b = float(self.V)

        try:
            self.c = inits['c']
        except KeyError:
            self.c = float(self.V)


        assert len(self.psi) == self.V

    def update(self, iterations=1000):

        for iteration in xrange(iterations):
            self._sample_bpsi()
            self._sample_c()

        return dict(psi = self.psi,
                    b = self.b,
                    c = self.c)

    def sample(self, number_of_samples=1000, thin=10):

        """
        Draw `number_of_samples` samples every `thin` steps.
        This will take `number_of_samples` x `thin` iterations.

        """

        psi = []
        b = []
        c = []

        for iteration in cycle(xrange(thin)):

            self._sample_bpsi()
            self._sample_c()

            if iteration == 0:
                psi.append(self.psi)
                b.append(self.b)
                c.append(self.c)

            if len(psi) >= number_of_samples:
                break


        return dict(psi=psi,
                    b = b,
                    c = c)

    # =============================================================

    def _sample_bpsi(self, seed=None):

        if seed is None:
            seed = randint(101, 1000001)

        I = unique(self.S)

        self.sigma_s_colsums, self.b, self.psi\
            = fortransamplers.polya_sampler_bpsi2(self.S, 
                                                  I, 
                                                  max(I), 
                                                  self.psi,
                                                  self.b, 
                                                  self.c, 
                                                  seed, 
                                                  len(I),
                                                  self.K, 
                                                  self.V)

    def _sample_c(self, seed=None):

        if seed is None:
            seed = randint(101, 1000001)

        I = unique(self.sigma_s_colsums)

        self.c = fortransamplers.polya_sampler_c2(self.sigma_s_colsums,  
                                                  I,
                                                  max(I),
                                                  self.c,
                                                  seed,
                                                  len(I),
                                                  self.V)
