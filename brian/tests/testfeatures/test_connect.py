from numpy.testing.utils import assert_equal

from brian import *

def test_delay_connect_with_subgroups():
    G = NeuronGroup(4, 'V:1')
    # test connect method
    C1 = Connection(G, G, 'V', delay=True, structure='dense')
    C2 = Connection(G, G, 'V', delay=True, structure='sparse')
    C3 = Connection(G, G, 'V', delay=True, structure='dynamic')
    for i, C in enumerate([C1, C2, C3]):
        C.connect(G[0:2], G[0:2], W=ones((2, 2)), delay=1)
        C.connect(G[0:2], G[2:4], W=ones((2, 2)), delay=(2.1, 2.2))
        C.connect(G[2:4], G[0:2], W=ones((2, 2)), delay=lambda:4)
        C.connect(G[2:4], G[2:4], W=ones((2, 2)), delay=lambda i, j:i + 10 * j)
        assert (C.W.todense() == 1).all(), 'Problem with connection C' + str(i + 1)
        D = C.delayvec.todense()
        assert (array(D, dtype=int) == array([[1, 1, 2, 2],
                                            [1, 1, 2, 2],
                                            [4, 4, 0, 10],
                                            [4, 4, 1, 11]], dtype=int)).all(), 'Problem with connection C' + str(i + 1)
    # test connect_random method
    C1 = Connection(G, G, 'V', delay=True, structure='dense')
    C2 = Connection(G, G, 'V', delay=True, structure='sparse')
    C3 = Connection(G, G, 'V', delay=True, structure='dynamic')
    for i, C in enumerate([C1, C2, C3]):
        C.connect_random(G[0:2], G[0:2], p=1, weight=1, delay=1)
        C.connect_random(G[0:2], G[2:4], p=1, weight=1, delay=(2.1, 2.2))
        C.connect_random(G[2:4], G[0:2], p=1, weight=1, delay=lambda:4)
        C.connect_random(G[2:4], G[2:4], p=1, weight=1, delay=lambda i, j:i + 10 * j)
        assert (C.W.todense() == 1).all(), 'Problem with connection C' + str(i + 1)
        D = C.delayvec.todense()
        assert (array(D, dtype=int) == array([[1, 1, 2, 2],
                                            [1, 1, 2, 2],
                                            [4, 4, 0, 10],
                                            [4, 4, 1, 11]], dtype=int)).all(), 'Problem with connection C' + str(i + 1)
    # test connect_full method
    C1 = Connection(G, G, 'V', delay=True, structure='dense')
    C2 = Connection(G, G, 'V', delay=True, structure='sparse')
    C3 = Connection(G, G, 'V', delay=True, structure='dynamic')
    for i, C in enumerate([C1, C2, C3]):
        C.connect_full(G[0:2], G[0:2], weight=1, delay=1)
        C.connect_full(G[0:2], G[2:4], weight=1, delay=(2.1, 2.2))
        C.connect_full(G[2:4], G[0:2], weight=1, delay=lambda:4)
        C.connect_full(G[2:4], G[2:4], weight=1, delay=lambda i, j:i + 10 * j)
        assert (C.W.todense() == 1).all(), 'Problem with connection C' + str(i + 1)
        D = C.delayvec.todense()
        assert (array(D, dtype=int) == array([[1, 1, 2, 2],
                                            [1, 1, 2, 2],
                                            [4, 4, 0, 10],
                                            [4, 4, 1, 11]], dtype=int)).all(), 'Problem with connection C' + str(i + 1)
    # test connect_one_to_one method
    C1 = Connection(G, G, 'V', delay=True, structure='dense')
    C2 = Connection(G, G, 'V', delay=True, structure='sparse')
    C3 = Connection(G, G, 'V', delay=True, structure='dynamic')
    for i, C in enumerate([C1, C2, C3]):
        C.connect_one_to_one(G[0:2], G[0:2], weight=1, delay=1)
        C.connect_one_to_one(G[0:2], G[2:4], weight=1, delay=(2.1, 2.2))
        C.connect_one_to_one(G[2:4], G[0:2], weight=1, delay=lambda:4)
        C.connect_one_to_one(G[2:4], G[2:4], weight=1, delay=lambda i, j:i + 10 * j)
        D = C.delayvec.todense()
        assert (array(D, dtype=int) == array([[1, 0, 2, 0],
                                            [0, 1, 0, 2],
                                            [4, 0, 0, 0],
                                            [0, 4, 0, 11]], dtype=int)).all(), 'Problem with connection C' + str(i + 1)


def test_connect_with_seed():
    G1 = NeuronGroup(10, 'v:1')
    G2 = NeuronGroup(10, 'v:1')
    C1 = Connection(G1, G2, 'v')
    C2 = Connection(G1, G2, 'v')
    # We check two things:
    # 1. Providing a seed to connect_random should produce exactly the same
    #    connection pattern every time
    # 2. This seed argument does not affect the general random number generation
    np.random.seed(12345)
    numbers = np.random.rand(5)
    np.random.seed(12345)
    C1.connect_random(p=0.5, seed=54321)
    C2.connect_random(p=0.5, seed=54321)
    assert_equal(C1.W.todense(), C2.W.todense())
    # The general random number generation should be unaffected and use the seed
    # specified above
    new_numbers = np.random.rand(5)
    assert_equal(numbers, new_numbers)


def test_connect_with_subgroups():
    G = NeuronGroup(4, 'V:1')
    # test connect method
    C1 = Connection(G, G, 'V', structure='dense')
    C2 = Connection(G, G, 'V', structure='sparse')
    C3 = Connection(G, G, 'V', structure='dynamic')
    for i, C in enumerate([C1, C2, C3]):
        C.connect(G[0:2], G[0:2], W=ones((2, 2)))
        C.connect(G[0:2], G[2:4], W=2 * ones((2, 2)))
        C.connect(G[2:4], G[0:2], W=4 * ones((2, 2)))
        C.connect(G[2:4], G[2:4], W=array([[0, 10], [1, 11]]))
        W = C.W.todense()
        assert (array(W, dtype=int) == array([[1, 1, 2, 2],
                                            [1, 1, 2, 2],
                                            [4, 4, 0, 10],
                                            [4, 4, 1, 11]], dtype=int)).all(), 'Problem with connection C' + str(i + 1)
    # test connect_random method
    C1 = Connection(G, G, 'V', structure='dense')
    C2 = Connection(G, G, 'V', structure='sparse')
    C3 = Connection(G, G, 'V', structure='dynamic')
    for i, C in enumerate([C1, C2, C3]):
        C.connect_random(G[0:2], G[0:2], p=1, weight=1)
        C.connect_random(G[2:4], G[2:4], p=1, weight=lambda i, j:i + 10 * j)
        W = C.W.todense()
        assert (array(W, dtype=int) == array([[1, 1, 0, 0],
                                            [1, 1, 0, 0],
                                            [0, 0, 0, 10],
                                            [0, 0, 1, 11]], dtype=int)).all(), 'Problem with connection C' + str(i + 1)
    # test connect_full method
    C1 = Connection(G, G, 'V', structure='dense')
    C2 = Connection(G, G, 'V', structure='sparse')
    C3 = Connection(G, G, 'V', structure='dynamic')
    for i, C in enumerate([C1, C2, C3]):
        C.connect_full(G[0:2], G[0:2], weight=1)
        C.connect_full(G[2:4], G[2:4], weight=lambda i, j:i + 10 * j)
        W = C.W.todense()
        assert (array(W, dtype=int) == array([[1, 1, 0, 0],
                                            [1, 1, 0, 0],
                                            [0, 0, 0, 10],
                                            [0, 0, 1, 11]], dtype=int)).all(), 'Problem with connection C' + str(i + 1)
    # test connect_one_to_one method
    C1 = Connection(G, G, 'V', structure='dense')
    C2 = Connection(G, G, 'V', structure='sparse')
    C3 = Connection(G, G, 'V', structure='dynamic')
    for i, C in enumerate([C1, C2, C3]):
        C.connect_one_to_one(G[0:2], G[0:2], weight=1)
        W = C.W.todense()
        assert (array(W, dtype=int) == array([[1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [0, 0, 0, 0],
                                            [0, 0, 0, 0]], dtype=int)).all(), 'Problem with connection C' + str(i + 1)

if __name__ == '__main__':
    test_delay_connect_with_subgroups()
    test_connect_with_seed()
    test_connect_with_subgroups()
