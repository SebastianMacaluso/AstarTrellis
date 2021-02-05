import operator as op
import time
import itertools
from functools import reduce
from heapq import heappop, heappush, heappushpop
from itertools import combinations
# from . import Ginkgo_likelihood_invM as likelihood
from AstarTrellis import Ginkgo_likelihood_invM as likelihood

import networkx as nx
import numpy as np
import wandb
from absl import logging
from scipy.sparse import lil_matrix

logging.set_verbosity(logging.INFO)


# https://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python
def ncr(n, r):
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom  # or / in Python 2


def k_random_2_cuts(elem, k=None, num_tries=lambda x: x * 100):
    if len(elem) < 2:
        return []
    if k is None:
        if len(elem) == 2:
            k = 1
        else:
            k = len(elem)
    # if k is too big, we'll never get there
    assert ncr(len(elem), 2) >= k
    cuts = []
    for _ in range(num_tries(k)):
        np.random.shuffle(elem)
        i = np.random.randint(len(elem) - 1)
        i += 1
        cut = (elem[:i], elem[i:])
        if cut not in cuts:
            cuts.append(cut)
            if len(cuts) == k:
                break
    return cuts


def sparsest_cut(vertices, edges, s, t):
    nxg = nx.nx.Graph()
    nxg.add_nodes_from(vertices)
    for k, v in edges.items():
        # k = tuple(k)
        nxg.add_edge(k[0], k[1], capacity=v)
    res = nx.minimum_cut(nxg, s, t)
    return res




def top_k_of_n_2_cuts(elem, graph, k=None, n=None, num_tries=lambda x: x * 10, all_pairs_max_size=None, search_cap_size=40):
    if len(elem) > search_cap_size:
        return []
    if len(elem) < 2:
        return []
    if all_pairs_max_size and len(elem) <= all_pairs_max_size:
        return all_two_partitions(elem)
    nck = ncr(len(elem), 2)
    if k is None:
        # if len(elem) == 2:
        #     k = 1
        # else:
        #     k = len(elem)
        k = min(nck, 1000)
    if n is None:
        n = min(nck, k * 10)

    # if k is too big, we'll never get there
    # if nck >= k:
    #     return []
    # if nck >= n:
    #     return []

    cuts = []
    graph.todok()
    # print('elem', elem)
    # print('graph', graph)
    # print('combinations(elem, 2)', list(combinations(elem, 2)))
    edges = {(n1, n2): graph[n1, n2] for (n1, n2) in combinations(elem, 2)}
    # print('edges', edges)
    for _ in range(num_tries(n)):
        s, t = np.random.choice(elem, 2, replace=False)
        value, cut = sparsest_cut(elem, edges, s, t)
        # print('cut', cut)
        cut = [i for i in cut]
        c = []
        for side in cut:
            side = list(i for i in side)
            # not sure if sort is needed...
            side.sort()
            # side = ''.join(side)
            c.append(side)
        # print('ccc', c)
        c.sort()
        c = tuple(c)
        # print('c', c)
        # print('cuts', cuts)
        if not (value, c) in cuts:
            cuts.append((value, c))
        if len(cuts) >= n:
            break
    cuts.sort()
    return [i for _, i in cuts][:k]


def all_two_partitions(elem):
    """ Create all L,R children of elem """
    if len(elem) < 2:
        return []
    special_elem, other_elems = elem[0], elem[1:]
    # Add 1st L,R pair
    cuts = [([special_elem], other_elems)]
    # Make all combinations
    for i in range(1, len(other_elems)):
        for el in combinations(other_elems, i):
            l_child = [special_elem] + list(el)
            r_child = [i for i in elem if i not in l_child]
            cuts.append((l_child, r_child))
    logging.debug('all_two_partitions of %s gives %s values', str(elem), len(cuts))
    return cuts



def get_complement_node(self, node, elements=None):
    #def node_complement(self, node, elements=None):
    if elements is None:
        elements = self.elements

    """Subtract elements in node"""
    elem = elements - node.elements
    # print("Elements after subtracting special element = ",elem)
    return self.elements_2_node.get(elem)

def get_nodes_containing_element(self, root, element):
    ##
    nodes = {}
    def get_parents(node):
        if node == root:
            return
        if set(node).issubset(set(root)):
            nodes.add((node, root - node))
            for parent in self.parents[self.elems2node[node]]:
                get_parents(parent)

    # e_node = self.elements_2_node[frozenset(element)]
    get_parents(element)
    nodes = list(nodes)
    return nodes



# https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)

#########################
class IterTrellis(object):

    def __init__(self, max_nodes, propagate_values_up):
        pClass_st = time.time()
        # children[i] gives the kids of node i
        self.nodes_explored = 0
        self.children = [[] for _ in range(max_nodes)]
        self.mapv = np.zeros(max_nodes, dtype=np.float32) # MAP vertices
        self.arg_mapv = [[] for _ in range(max_nodes)] # Children of mapv
        self.clusters = [[] for _ in range(max_nodes)]  # list of frozensets
        # pq[i] = f, g, h, a, b
        self.pq = [[] for _ in range(max_nodes)]  # List of priority queues where each entry index is the node id.
        self.explored = np.zeros(max_nodes, dtype=np.bool_)
        self.elems2node = dict() #Nodes are indices and not a class like in the standard full trellis
        self.root = None
        self.next_id = 0
        self.last_memoized = -1 * np.ones(max_nodes, dtype=np.float32)
        self.current_extension_num = -1
        self.propagate_values_up = propagate_values_up
        self.MAP_f=np.inf
        self.MAP_hc = None
        if self.propagate_values_up:
            # parents[i] gives parents of node i
            self.parents = [[] for _ in range(max_nodes)]
            self.up_to_date_fgh = [{} for _ in range(max_nodes)]  # [i] -> {(c_l, c_r) -> f,g,h}
            self.values_on_q = [{} for _ in range(max_nodes)]  # [i] -> {(c_l, c_r) -> queue([f_j,g_j,h_j,...])}

        pClass_en_t = time.time()- pClass_st
        logging.info('Loading Parent class time: %s', pClass_en_t)



    def load_from_tree(self, tree, hc):
        """Not implemented for Ginkgo Jets"""
        # for i in self.__dict__.items():
        #     print('XXX: %s: %s' % i)
        # print()

        ## remove existing parameters and then replace them from tree. not LEAVE IN PLACE graph and _get_children
        IterTrellis.__init__(self, propagate_values_up=self.propagate_values_up, max_nodes=len(self.children))

        # for i in self.__dict__.items():
        #     print('%s: %s' % i)
        # print()

        self.next_id = tree.next_node_id
        ## set children
        for i, ch in enumerate(tree.children):
            if not ch:
                continue
            self.children[i].append(tuple(ch))
        if self.propagate_values_up:
            for i, child_list in enumerate(self.children):
                if not child_list:
                    continue
                # print('jjj', child_list)
                for ch_l, ch_r in child_list:
                    self.parents[ch_l].append(i)
                    self.parents[ch_r].append(i)

        # print(len(self.children))
        # print(type(self.children[0]))
        # print(self.children[0])
        # print()

        ## set clusters
        for i, cl in enumerate(tree.descendants):
            if not cl:
                continue
            self.clusters[i] = frozenset(cl)
        # print(len(self.clusters))
        # print(type(self.clusters[0]))
        # print(self.clusters[0])
        # print([i for i in self.clusters if i])

        ## set root
        root_id = None
        max_len_elem = 0
        for i, cl in enumerate(self.clusters):
            if cl:
                self.elems2node[cl] = i
                if len(cl) > max_len_elem:
                    root_id = i
                    max_len_elem = len(cl)

        assert root_id is not None
        self.root = root_id

        ## sort nodes from small to large (based on # of elements)
        ## for each node:
        # ## find node's children
        # ## get g's from children and psi function, set h = 0
        # ## set pq, mapv, argmapv (mapv is the map vertex and argmapv saves the children of mapv)
        for elems in sorted(self.elems2node, key=lambda k: len(k)):
            node = self.elems2node[elems]
            # print(elems, node)
            if len(elems) == 1:
                self.mapv[node] = 0
                self.arg_mapv[node] = node
                continue
            # print('elems', elems)
            # print('node', node)
            # print('children', self.children[node])
            self.explored[node] = True
            ch_l, ch_r = self.children[node][0]
            self.mapv[node] = self.g_fun(ch_l, ch_r) + self.mapv[ch_l] + self.mapv[ch_r]
            self.arg_mapv[node] = [ch_l, ch_r]
            # heappush(self.pq[node], (self.mapv[node], self.mapv[node], 0, ch_l, ch_r))
            self.push(node, (self.mapv[node], self.mapv[node], 0, ch_l, ch_r))

    def push(self, i, v):
        f, g, h, ch_l, ch_r = v
        #
        if not self.propagate_values_up:
            return heappush(self.pq[i], (f, g, h, ch_l, ch_r))

        chs = frozenset((ch_l, ch_r))
        q_vals = self.values_on_q[i].get(chs)

        def do_push():
            heappush(self.pq[i], (f, g, h, ch_l, ch_r))
            if not q_vals:
                self.values_on_q[i][chs] = []
            # heappush(self.values_on_q[i][chs], (f, g, h, ch_l, ch_r))
            heappush(self.values_on_q[i][chs], (f, g, h))

        # update the up-to-date hash with f, g, h.
        self.up_to_date_fgh[i][chs] = (f, g, h)
        # if f < f'', where f'' is the min value for ch_l, ch_r currently present on queue,
        # push (f, g, h, ch_l, ch_r) onto priority queue (and hash of queues), otherwise do not.

        if not q_vals:
            return do_push()
        # print('top_of_q_vals', q_vals[0])
        min_f, min_g, min_h = q_vals[0]
        if f < min_f:
            # do push
            return do_push()
        ## else do not push

    def pop(self, i):
        # Pop from the priority queue, from entry i
        # if i == 17:
        #     print(self.pq[i])
        f, g, h, ch_l, ch_r = heappop(self.pq[i])
        #
        if not self.propagate_values_up:
            return f, g, h, ch_l, ch_r
        # (and hash of queues).
        chs = frozenset((ch_l, ch_r))
        heappop(self.values_on_q[i][chs])
        # Get up-to-date hash value f', g', h' for (ch_l, ch_r).
        f_, g_, h_ = self.up_to_date_fgh[i][chs]
        # If** the up-to-date value f' is less than the queue value f, return f', g', h', ch_l, ch_r.
        # ** I think we will maintain an invariant that this case should never be encountered...
        assert f_ >= f
        # If f is equal to f', return f', g', h', ch_l, ch_r.
        if f == f_:
            return f_, g_, h_, ch_l, ch_r
        # If the up-to-date f' is greater than queue value f, we call push
        # on f',g',h',ch_l,ch_r and retry pop.
        self.push(i, (f_, g_, h_, ch_l, ch_r))
        return self.pop(i)

    def peek(self, i):
        if i == 17:
            print('peeking', i, self.pq[i])
        v = self.pop(i)
        self.push(i, v)
        return v

    def g_fun(self, ch_l, ch_r):
        """Return the g value for a split of ch_l and ch_r

        :param ch_l: left child
        :param ch_r: right child
        :return:
        """
        return 0.0

    def h_fun(self, ch_i, sib_i=None):
        """Provide an estimate of the cost, h.

        :param ch_i: the subtree root.
        :return:
        """
        return np.Inf

    def set_root(self, elems):
        self.root = self.record_node(elems)
        self.clusters[self.root] = elems
        self.nodes_explored += int(self.root)  # Add the number of elements to the nodes explored

    def is_leaf(self, i):
        """ Returns true if i is a leaf.

        True if i has no children

        :param i: node id
        :return: if there are no children.
        """
        return len(self.children[i]) == 0 or len(self.pq[i]) == 0

    def get_state(self, all_pairs_max_size=2, num_tries=4):
        """Return the best available (partial) hierarchical clustering.

        :return: hc - hierarchical clustering
        :return: internals - internal nodes
        :return: lvs - leaf nodes
        :return: parent2children - parent2children map
        """
        logging.debug('Getting current state.')
        # Start from the root
        travel = [self.root]
        hc = []
        internals = []
        lvs = []
        parent2children = dict()
        while travel:
            i = travel.pop()
            hc.append(i)
            logging.debug('visiting node %s', i)

            # if self.is_leaf(i): # leaf at current state
            if (len(self.pq[i])<2 and len(self.clusters[i])>2 ) or self.is_leaf(i):  # leaf at current state (we do this because we already added beam search nodes)

                logging.debug('visiting node %s - is leaf', i)
                lvs.append(i)
                elements = self.clusters[i]
                # assert len(elements) > 0
                if len(elements) > 1:
                    # if self.is_leaf(i):
                    self.nodes_explored += 1
                    self.initialize(i, elements, all_pairs_max_size=all_pairs_max_size, num_tries=num_tries)
            else:
                logging.debug('visiting node %s - is iternal', i)
                internals.append(i)
            if self.explored[i]:
                a, b = self.walk_down(i)
                logging.debug('visiting node %s - has kids %s and %s', i, a, b)
                parent2children[i] = [a, b]
                self.memoize_if_needed(i)
                travel.append(a)
                travel.append(b)

        return hc, internals, lvs, parent2children

    def memoize_if_needed(self, i):
        """Memoize the values of i if we know them.

        :param i: node id
        :return: Nothing.
        """
        f, g, h, a, b = self.pq[i][0]
        if (h == 0.0 and self.last_memoized[a] == self.current_extension_num
          and self.last_memoized[b] == self.current_extension_num):
            logging.debug('memoizing node %s - g value %s - children %s and %s', i, g, a, b)
            self.mapv[i] = g
            self.arg_mapv[i] = [a, b]
            self.last_memoized[i] = self.current_extension_num

    def is_goal_state(self, hc, internals, lvs):
        """Return true if this is a goal state.

        In this setting, if all of the leaves of the tree
        are singleton clusters.

        :param hc: the full tree
        :param internals: the internal nodes
        :param lvs: the leaves
        :return: true if this is a goal state.
        """
        return all([len(self.clusters[x]) == 1 for x in lvs])

    def walk_down(self, i):
        """Return the children of i in the current state of the tree.

        If i has a map value already, return these children.
        Otherwise pull something off the frontier.

        :param i: node i
        :return: the children of i
        """
        if self.arg_mapv[i] and self.last_memoized[i] == self.current_extension_num:
            logging.debug('walking down %s hit mapv', i)
            return self.arg_mapv[i]
        else:
            f, g, h, a, b = self.pq[i][0]
            return a, b

    def record_node(self, elements: frozenset) -> int:
        """Get the node corresponding to the given elements, create new id if needed.

        Creates a new id if needed.

        :param elements: the elements (FrozenSet of Integers)
        :return: the node for the given elements
        """
        logging.debug('get node id from elements %s', str(elements))
        if elements not in self.elems2node:
            logging.debug('get node id from elements %s. new node! %s', str(elements), self.next_id)
            self.elems2node[elements] = self.next_id
            self.clusters[self.next_id] = elements
            self.next_id += 1
            return self.next_id - 1
        else:
            return self.elems2node[elements]

    # def _get_children(elems):
    #     raise NotImplementedError

    # def _get_children(self, root):
    #     ##
    #     root= frozenset(root)
    #     other_element = frozenset(list(root)[1::])
    #     # logging.info("other_elements = %s", other_element)
    #     nodes = []
    #
    #     for i in range(1,len(other_element)+1):
    #         nodes+=[(list(frozenset(elems)), list(root - frozenset(elems))) for elems in itertools.combinations(other_element, i ) ]
    #
    #     # def get_parents(node):
    #     #     logging.info("root  = %s | node = %s", root, node)
    #     #     if node == root:
    #     #         return
    #     #     if set(node).issubset(set(root)):
    #     #         nodes.add((node, root - node))
    #     #         for parent in self.parents[self.elems2node[node]]:
    #     #             get_parents(parent)
    #     #
    #     # # e_node = self.elements_2_node[frozenset(element)]
    #     # get_parents(special_element)
    #     # nodes = list(nodes)
    #     # logging.info("Left/right pairs = %s", nodes)
    #     return nodes

    def _get_children(self, root):
        ##
        root= frozenset(root)
        other_element = frozenset(list(root)[1::])
        nodes = []

        for i in range(1,len(other_element)+1):
            nodes+=[(self.record_node(frozenset(elems)), self.record_node(root - frozenset(elems))) for elems in itertools.combinations(other_element, i ) ]

        return nodes


    def Add_BeamSearchHC(self,
                   jet,
                   node_id,
                   root):
        """Add inner nodes and hierarchical structure of each input tree to the trellis graph. For each inner node find the leaves ids for the subtree rooted at its left child (this is the label of the node) and create left and right children nodes."""

        if jet["tree"][node_id, 0] != -1:

            L_child = jet["tree"][node_id, 0]
            """Find the leaves for the subtree rooted at itself"""
            outers = []
            LchildLeaves = self.leavesIDX(
                jet,
                L_child,
                outers,
            )

            # logger.debug(f"LchildLeaves = {LchildLeaves}")
            """pT ordered leaves to element"""

            elem_l = frozenset([self.id_from_momentum[str(entry)] for entry in LchildLeaves])
            elem_r= root - elem_l
            ch_l= self.record_node(elem_l)
            ch_r=self.record_node(elem_r)

            logging.debug("L elements = %s", elem_l)
            logging.debug("R elements = %s", elem_r)
        # for ch_l, ch_r in ch:
            g = self.g_fun(ch_l, ch_r)
            h = 0
            h += self.h_fun(ch_l)
            h += self.h_fun(ch_r)
            # h = 0
            # logging.debug('initialize node %s with children %s and %s | f %s | g %s | h %s', i, ch_l, ch_r, g + h, g, h)
            # for ch_i in [ch_l, ch_r]:
            #     # if the child is already there, grab values from it
            #     if self.explored[ch_i]:
            #         _, g_i, h_i, _, _ = self.pq[ch_i][0]
            #         g += g_i
            #         h += h_i
            #     else:
            #         h += self.h_fun(ch_i)
            #     logging.debug('Update of initialize node %s with children %s and %s | f %s | g %s | h %s', i, ch_l, ch_r, g + h, g, h)
                # if len(elems)==2 and h!=0:
            #     logging.info('+++++++++')
            #     logging.info('h_i = %s , h = %s, ch = %s, elem = %s',str(self.h_fun(ch_i)),str(h), str(ch_i), str(self.clusters[ch_i]))
            # logging.info('+=+=+=+=+=+=+=+=')

            # self.explored[i] = True  # Should this be outside the for loop? SM
            # heappush(self.pq[i], (g + h, g, h, ch_l, ch_r))
            self.push(self.elems2node[root], (g + h, g, h, ch_l, ch_r))

            self.Add_BeamSearchHC(
                jet,
                jet["tree"][node_id, 0],
                elem_l,
            )

            self.Add_BeamSearchHC(
                jet,
                jet["tree"][node_id, 1],
                elem_r,
            )




    def initialize(self, i, elems, all_pairs_max_size=2, num_tries=4):
        """Initialize the node i.

        This explores the node i. It sets the priority queue of i accordingly. If len(elems)==1, doe not do anything

        :param i: the node i
        :param elems: the elements at the node
        :return:

        """
        # logging.info('-------')
        # logging.info('node = %s, elem=%s', str(i),str(elems))
        # logging.info('-------')
        ch = self.get_children(i, elems, all_pairs_max_size=all_pairs_max_size, num_tries=num_tries)
        logging.debug('initialize node %s with %s children', i, len(ch))

        # t_init = time.time()
        # logging.info("Children = %s", ch)
        for ch_l, ch_r in ch:
            g = self.g_fun(ch_l, ch_r)
            h = 0
            logging.debug('initialize node %s with children %s and %s | f %s | g %s | h %s', i, ch_l, ch_r, g + h, g, h)
            for ch_i in [ch_l, ch_r]:
                # if the child is already there, grab values from it
                if self.explored[ch_i]:
                    _, g_i, h_i, _, _ = self.pq[ch_i][0]
                    g += g_i
                    h += h_i
                else:
                    h += self.h_fun(ch_i)
                logging.debug('Update of initialize node %s with children %s and %s | f %s | g %s | h %s', i, ch_l, ch_r, g + h, g, h)
                # if len(elems)==2 and h!=0:
            #     logging.info('+++++++++')
            #     logging.info('h_i = %s , h = %s, ch = %s, elem = %s',str(self.h_fun(ch_i)),str(h), str(ch_i), str(self.clusters[ch_i]))
            # logging.info('+=+=+=+=+=+=+=+=')

            # self.explored[i] = True  # Should this be outside the for loop? SM
            # heappush(self.pq[i], (g + h, g, h, ch_l, ch_r))
            self.push(i, (g + h, g, h, ch_l, ch_r))

        # logging.info("Init time = %s", time.time()- t_init)
        if len(elems)>1: #We can't set the leaves as explored, as we don't initialize a pq for them
            self.explored[i] = True  # Should this be outside the for loop? SM

        # return time.time()- t_init


    def get_children(self, i, elems, all_pairs_max_size=2, num_tries = 4):
        """Gives the children of node i that has elements elems.

        In this version, it grabs all 2 partitions if they are not there
        and caches this in children[i].

        :param i: The node i
        :param elems: The elements associated with node i
        :return: the children of i
        """
        if len(elems) == 1:
            return []
        elif self.explored[i]:
            return self.children[i]
        else:
            # self.children[i] = self._get_children(list(elems))  # all_two_partitions(list(elems))
            self.children[i] = self._top_k_children(list(elems),all_pairs_max_size=all_pairs_max_size, num_tries=num_tries)

                    # self.update_from_children(i, (ch_l, ch_r))
            return self.children[i]


    def _top_k_children(self, root, all_pairs_max_size=2, num_tries=10):
        """ Sample children and keep the top k that maximize the splitting likelihood"""
        root = frozenset(root)
        if len(root) <= all_pairs_max_size:
            return self._get_children(root)

        else:
            return self._sample_children(root, all_pairs_max_size=all_pairs_max_size, num_tries=num_tries)


    def _sample_children(self, root, all_pairs_max_size=2, num_tries=10):
        pass



    def update_from_children(self, p, children, hc=None, is_goal_state=False):
        """Update the parents priority queue.

        :param p: the parent id
        :param children: list (of size 2) of the children ids
        :return:
        """
        logging.debug('update %s from the children %s', p, str(children))
        logging.debug('-------==============-------')
        # if p == self.root and is_goal_state:
        #     logging.info('Before update p=%s | f,g,h,ch_l,ch_r =%s  ',
        #              p,  str(self.pq[p][0]))
        # print('update %s from the children %s' % (p, str(children)))
        ch_l, ch_r = children
        g = self.g_fun(ch_l, ch_r)
        h = 0
        for ch_i in [ch_l, ch_r]:
            # if the child is already there, grab values from it
            if self.explored[ch_i]:
                _, g_i, h_i, l, r = self.pq[ch_i][0]
                # print('ch_i, cl_ch_i, g_i, h_i', ch_i, self.clusters[ch_i], g_i, h_i)
                g += g_i
                h += h_i

                # if h_i!=0 and is_goal_state:
                #     """This is possible, because whe we update the pqs of previous vertices, unexplored paths become the 1 entry of the pq"""
                #     logging.info("H value for inners = %s | children vertices = %s and %s",h_i, self.clusters[l], self.clusters[r])
            else:
                # print('ch_i, cl_ch_i, hfun_i', ch_i, self.clusters[ch_i], self.h_fun(ch_i))
                h += self.h_fun(ch_i)
                # if self.h_fun(ch_i)!=0 and is_goal_state:
                #     logging.info("H value for leaves = %s", self.h_fun(ch_i))

        # heappop(self.pq[p])
        # heappush(self.pq[p], (g + h, g, h, ch_l, ch_r))
        self.pop(p)
        # print('pushing p, cl_p, f, g, h, cl, cr', p, self.clusters[p], g + h, g, h, ch_l, ch_r)
        # print()
        logging.debug('p=%s | f=%s | g=%s | h=%s | ch_l=%s | ch_r=%s ',
                     p,  g+h, g, h, ch_l, ch_r)
        logging.debug('-------==============-------')
        self.push(p, (g + h, g, h, ch_l, ch_r))

        if p==self.root:
            logging.debug("Updated value for root node = %s", g+h)
            logging.debug("===++++====" * 10)
        if p==self.root and h==0 and is_goal_state and self.MAP_f > g+h:
            # logging.info("root %s | p = %s | is goal state = %s",self.root,p, is_goal_state)
            logging.debug("New MAP value for root node g = %s | h=%s", g,h)
            logging.debug("===++++===="*10)
            self.MAP_f = g+h
            self.MAP_hc = hc

    def _execute_search(self, max_steps=np.Inf, all_pairs_max_size=2, num_tries=4):
        """Run A* Search.
        We aren't using max_steps ?

        :param max_steps:
        :return: hc, f - the clustering and f values
        """
        logging.info('Running A* Search. max_steps = %s', max_steps)
        step = 0
        most_leaves = 0
        time_sum = 0
        time_sum1 =0
        time_sum2 = 0
        time_sum3 = 0
        # self.MAP_f=np.inf
        # self.MAP_hc = None

        while (step<max_steps):
            st_t = time.time()

            # 1. Get the partial hierarchical clustering
            if len(self.pq[self.root]) > 0:
                f, g, h, _, _ = self.pq[self.root][0]
            else:
                f = -1  # something that is not zero
                g = -1  # something that is not zero
                h = -1  # something that is not zero

            hc, internals, lvs, parent2child = self.get_state(all_pairs_max_size=all_pairs_max_size, num_tries=num_tries)
            is_goal_state = self.is_goal_state(hc, internals, lvs)
            logging.debug("-------------------------------------------")



            if is_goal_state and h != 0:
                logging.debug('At goal state, but h is not zero!')
                logging.debug('leaves:\n%s', lvs)
                logging.debug('hc:\n%s', '\n'.join([str((x, self.clusters[x])) for x in hc]))
                logging.debug('Step=%s | num_leaves=%s | max_leaves_so_far=%s | f=%s | g=%s | h=%s',
                             step, len(lvs),
                             most_leaves, f, g, h)



                # print()
                # for x in hc:
                #     try:
                #         q = self.pq[x][0]
                #     except IndexError:
                #         q = 'N\A'
                #     #print(x, self.clusters[x], q)
                # print()
                # return None, None
                # print('hc:\n%s' % '\n'.join([str((x, self.clusters[x])) for x in hc]))
                # print('g,h:\n%s' % '\n'.join([str((self.pq[x][0])) for x in hc]))

            if is_goal_state and h == 0:
                logging.info('Reached goal state!')

                wandb.log({'num_leaves': len(lvs), 'max_leaves_so_far': most_leaves,
                           'total_time': time_sum, 'avg_time_per_step': time_sum / (step + 1),
                           'steps': step})


                # logging.debug('leaves: %s', str(lvs))
                # print('top of root:\n%s' % '\n'.join([str(len(self.pq[self.root])) for i in range(10)]))
                # print('top of root:\n%s' % '\n'.join([str(self.pq[self.root][i]) for i in range(min(10, len(self.pq[self.root])))]))

                # print('hc', hc)
                # print('lhc', len(hc))
                # print('thc', [type(i) for i in hc])
                # print('hcXXX:\n%s' % '\n'.join([str(self.clusters[x]) for x in hc]))
                # print('hcXXXHC:\nX%s' % '\nX'.join([str(x) for x in hc]))
                # print('hcXXXHCZZ:\nY%s' % '\nY'.join([str(self.clusters[x]) for x in hc]))

                # print('hcXXXHC: %s' % '\n'.join(hc))

                logging.info('hc:\n%s', '\n'.join([str((x, self.clusters[x])) for x in hc]))
                logging.info('Step=%s | num_leaves=%s | max_leaves_so_far=%s | f=%s | g=%s | h=%s',
                             (step+1), len(lvs),
                             most_leaves, f, g, h)
                # print('171717', self.clusters[17], self.pq[17], self.children[17])
                assert f == g
                assert h == 0
                return hc, f, step
            most_leaves = max(len(lvs), most_leaves)

            st_t1 = time.time()
            # # 2. Explore the leaves and initialize them. If len(elements)==1 then do not do anything
            # for l in lvs:
            #     # logging.info('Exploring leaves: %s', str(lvs))
            #     elements = self.clusters[l]
            #     assert len(elements) > 0
            #     if len(elements) > 1:
            #         self.initialize(l, elements)

            st_t2 = time.time()

            # 3. Update the internals. We update level by level, adding the children value of g to the parent? And then moving to the next level up.
            for i in range(len(internals) - 1, -1, -1):
                # if the node is not a leaf node in the partial hierarchical clustering
                if internals[i] in parent2child:
                    # if i==0: logging.info('internals[0]: %s', str(internals[0]))
                    self.update_from_children(internals[i], parent2child[internals[i]], hc=hc, is_goal_state=is_goal_state)

            # updated_f, g, h, _, _ = self.pq[self.root][0]
            # if updated_f< MAP_f:
            #     MAP_hc, MAP_f = hc, updated_f

            # logging.
            en_t = time.time()
            time_sum += en_t - st_t
            time_sum1 += st_t1 - st_t
            time_sum2 += st_t2- st_t1
            time_sum3 += en_t - st_t2

            logging.log_every_n(logging.DEBUG,
                                'Step=%s | num_nodes=%s | num_internals=%s | num_leaves=%s | max_leaves_so_far=%s | f=%s | g=%s | h=%s',
                                1, step, len(hc), len(internals), len(lvs),
                                most_leaves, f, g, h)
            if step % 100 == 0:
                wandb.log({'num_leaves': len(lvs), 'max_leaves_so_far': most_leaves,
                           'total_time': time_sum, 'step_time': en_t - st_t, 'avg_time_per_step': time_sum / (step + 1),
                           'steps': step})
                # logging.info('steps= %s | total_time = %s | step_time = %s | avg_time_per_step = %s |  t1=%s | t2 = %s | t3 =%s', step, time_sum, en_t - st_t, time_sum / (step + 1), st_t1 - st_t, st_t2-st_t1, en_t-st_t2)
            #     logging.info(
            #         'steps= %s | total_time = %s | avg_time_per_step so far = %s |  avg t1=%s | avg t2 = %s | avgt3 =%s',
            #         step, time_sum, time_sum / (step + 1), time_sum1/ (step + 1), time_sum2/ (step + 1) , time_sum3/ (step + 1))
                logging.info(
                    'steps= %s | total_time = %s | avg_time_per_step so far = %s |  avg t1=%s | avg t2 = %s | avgt3 =%s',
                    step, time_sum, time_sum / (step + 1), time_sum1 , time_sum2 ,
                                    time_sum3 )

            step += 1
        else:
            return self.MAP_hc, self.MAP_f, step


    def extend(self, i, elems):
        """Extend the node i.

        This extends the queue the node i.
        It sets the priority queue of i accordingly.
        Add new entries to the queue of i for each pair of children?

        :param i: the node i
        :param elems: the elements at the node
        :return: children added to the queue (possibly already on the queue)
        """
        ch = self.get_children(i, list(elems))
        logging.debug('extend node %s with %s children', i, len(ch))
        # print('extending node %s: has %s children, extending with %s children' % (i, len(self.children[i]), len(ch)))

        for ch_l, ch_r in ch:
            # for l, r in ch:
            #     ch_l = self.record_node(frozenset((l,)))
            #     ch_r = self.record_node(frozenset((r,)))
            #     # if (ch_l, ch_r) in self.children[i] or (ch_r, ch_l) in self.children[i]:
            #     #     # print('found existing child')
            #     #     pass
            #     # else:
            #     #     self.children[i].append((ch_l, ch_r))
            #     if self.propagate_values_up:
            #         if not i in self.parents[ch_l]:
            #             self.parents[ch_l].append(i)
            #         if not i in self.parents[ch_r]:
            #             self.parents[ch_r].append(i)
            g = self.g_fun(ch_l, ch_r)
            h = 0
            # print('extending node %s with children %s and %s | g %s | h %s' % (i, ch_l, ch_r, g, h))
            logging.debug('extending node %s with children %s and %s | g %s | h %s', i, ch_l, ch_r, g, h)
            for ch_i in [ch_l, ch_r]:
                # if the child is already there, grab values from it
                if self.explored[ch_i]:
                    _, g_i, h_i, _, _ = self.peek(ch_i)
                    g += g_i
                    h += h_i
                else:
                    h += self.h_fun(ch_i)
                # print('Extending node %s with children %s and %s | g %s | h %s' % (i, ch_l, ch_r, g, h))
                logging.debug('extending node %s with children %s and %s | g %s | h %s', i, ch_l, ch_r, g, h)
            self.explored[i] = True
            self.push(i, (g + h, g, h, ch_l, ch_r))
        return ch

    def scaffold_from_tree(self, tree, hc, propagate_values_up):
        ## remove existing parameters and then replace them from tree. not LEAVE IN PLACE graph and _get_children
        IterTrellis.__init__(self, max_nodes=len(self.children), propagate_values_up=propagate_values_up)

        self.next_id = tree.next_node_id

        # set children
        for i, ch in enumerate(tree.children):
            if i > tree.next_node_id:
                break
            if not ch:
                continue
            self.children[i].append(tuple(ch))
            if self.propagate_values_up:
                for c in ch:
                    self.parents[c].append(i)
            # self.explored[i] = True

        # set clusters
        for i, cl in enumerate(tree.descendants):
            if not cl:
                continue
            self.clusters[i] = frozenset(cl)
            if i > tree.next_node_id:
                break

        # set root
        root_id = None
        max_len_elem = 0
        for i, cl in enumerate(self.clusters):
            if cl:
                self.elems2node[cl] = i
                if len(cl) > max_len_elem:
                    root_id = i
                    max_len_elem = len(cl)
        assert root_id is not None
        self.root = root_id

        # for i, elems in enumerate(self.clusters):
        #     if elems:
        #         self.elems2node[frozenset(elems)] = i

        # # sort nodes from small to large (based on # of elements)
        # # for each node:
        # ## find node's children
        # ## get g's from children and psi function, set h = 0
        # ## set pq, mapv, argmapv
        # for elems in sorted(self.elems2node, key=lambda k: len(k)):
        #     if not elems:
        #         continue
        #     i = self.elems2node[elems]
        #     #print(elems, node)
        #     #if len(elems) == 1:
        #     if not self.children[i]:
        #         continue
        #     for ch_l, ch_r in self.children[i]:
        #         if not self.children[ch_l]:
        #             g_l = h_l = 0
        #         else:
        #             _, g_l, h_l, _, _ = self.peek(ch_l)
        #         if not self.children[ch_r]:
        #             g_r = h_r = 0
        #         else:
        #             _, g_r, h_r, _, _ = self.peek(ch_r)
        #         g = self.g_fun(ch_l, ch_r) + g_l + g_r
        #         h = h_l + h_r
        #         f = g + h
        #         self.push(i, (f, g, h, ch_l, ch_r))

        #     if -1 in elems:
        #         # self.mapv[node] = 0
        #         # self.arg_mapv[node] = node
        #         continue
        #     # print('elems', elems)
        #     # print('node', node)
        #     # print('children', self.children[node])
        #     #self.explored[node] = True
        #     ch_l, ch_r = self.children[node][0]
        #     g = self.g_fun(ch_l, ch_r)
        #     h = self.h_fun(ch_l,ch_r) + self.h_fun(ch_r,ch_l)
        #     self.arg_mapv[node] = [ch_l, ch_r]
        #     # heappush(self.pq[node], (self.mapv[node], self.mapv[node], 0, ch_l, ch_r))
        #     self.push(node, (g+h, g, 0, ch_l, ch_r))

    def reset(self):
        self.mapv = np.zeros(len(self.mapv), dtype=np.float32)
        self.arg_mapv = [[] for _ in range((len(self.arg_mapv)))]
        self.pq = [[] for _ in range(len(self.pq))]
        self.explored = np.zeros(len(self.explored), dtype=np.bool_)

    def extend_hc_alt(self):
        hc, internals, lvs, parent2child = self.get_state()
        self.reset()
        # for c in hc:
        #    logging.debug('unmemoizing node %s', c)
        #    self.extend_alt(c, self.clusters[c])

    def extend_alt(self, i, elems):
        """Extend the node i.

        This extends the queue the node i.
        It sets the priority queue of i accordingly.

        :param i: the node i
        :param elems: the elements at the node
        :return: children added to the queue (possibly already on the queue)
        """
        ch = self.get_children(i, list(elems))
        logging.debug('extend node %s with %s children', i, len(ch))
        # print('extending node %s: has %s children, extending with %s children' % (i, len(self.children[i]), len(ch)))

        # for ch_l, ch_r in ch:
        #     # for l, r in ch:
        #     #     ch_l = self.record_node(frozenset((l,)))
        #     #     ch_r = self.record_node(frozenset((r,)))
        #     #     # if (ch_l, ch_r) in self.children[i] or (ch_r, ch_l) in self.children[i]:
        #     #     #     # print('found existing child')
        #     #     #     pass
        #     #     # else:
        #     #     #     self.children[i].append((ch_l, ch_r))
        #     #     if self.propagate_values_up:
        #     #         if not i in self.parents[ch_l]:
        #     #             self.parents[ch_l].append(i)
        #     #         if not i in self.parents[ch_r]:
        #     #             self.parents[ch_r].append(i)
        #     g = self.g_fun(ch_l, ch_r)
        #     h = 0
        #     # print('extending node %s with children %s and %s | g %s | h %s' % (i, ch_l, ch_r, g, h))
        #     logging.debug('extending node %s with children %s and %s | g %s | h %s', i, ch_l, ch_r, g, h)
        #     for ch_i in [ch_l, ch_r]:
        #         # if the child is already there, grab values from it
        #         if self.explored[ch_i]:
        #             _, g_i, h_i, _, _ = self.peek(ch_i)
        #             g += g_i
        #             h += h_i
        #         else:
        #             h += self.h_fun(ch_i)
        #         # print('Extending node %s with children %s and %s | g %s | h %s' % (i, ch_l, ch_r, g, h))
        #         logging.debug('extending node %s with children %s and %s | g %s | h %s', i, ch_l, ch_r, g, h)
        #     self.explored[i] = True
        #     self.push(i, (g + h, g, h, ch_l, ch_r))
        # return ch

    def extend_hc(self):
        hc, internals, lvs, parent2child = self.get_state()
        # print('hc....', hc)
        # print('internals', internals)
        # print('leaves', lvs)
        # print('root', self.root)

        ## TODO: remove the below and reset maps only for necessary nodes
        # self.mapv = np.zeros(len(self.mapv), dtype=np.float32)
        # self.arg_mapv = [[] for _ in range(len(self.arg_mapv))]

        nodes_to_update = hc
        nodes_to_update.sort(reverse=False, key=lambda x: len(self.clusters[x]))
        for c in nodes_to_update:
            logging.debug('unmemoizing node %s', c)
            # self.mapv[c] = 0
            # self.arg_mapv[c] = []
            self.extend(c, self.clusters[c])
        # print("XXX tending hc")
        # print('pu', self.propagate_values_up)
        if self.propagate_values_up:
            #
            # When updating node n during a trellis expansion, we update the up-to-date dict of every parent of n,
            # denoted, p. Since the updates during a trellis expansion will always be a reduction in cost, we also
            # push the reduced f,g,h values onto the priority queue (and hash of queues) at p. If the update changes
            # the min value of the priority queue at p, then p is added to the list of nodes that need to update their
            # parents.
            #
            nodes_to_update = hc
            for i, child_list in enumerate(self.children):
                for ch_l, ch_r in child_list:
                    assert i in self.parents[ch_l]
                    assert i in self.parents[ch_r]
            while nodes_to_update:
                nodes_to_update.sort(reverse=False, key=lambda x: len(self.clusters[x]))
                ch_l = nodes_to_update.pop()
                # print('>>>>>>>>>>>>>>>>>')
                print('updating ch_l', self.clusters[ch_l])
                # print('parents of ch_l', [self.clusters[x] for x in self.parents[ch_l]])
                # print('children of parents', [(self.clusters[x], [(self.clusters[y1], self.clusters[y2]) for y1, y2 in self.children[x]]) for x in self.parents[ch_l]])
                # self.mapv[i] = 0
                # self.arg_mapv[i] = []
                # print('pqii', self.clusters[ch_l])
                if len(self.clusters[ch_l]) > 1:
                    # print('chl', self.clusters[ch_l])
                    # print('pq[ch_l]', self.peek(ch_l))
                    # print('self.up_to_date_fgh[ch_l]', self.up_to_date_fgh[ch_l])
                    _, l_g, l_h, _, _ = self.peek(ch_l)
                else:
                    l_g = l_h = 0
                for p in self.parents[ch_l]:
                    print('dealing with parent (%s) of child(%s)' % (self.clusters[p], self.clusters[ch_l]))
                    # self.mapv[p] = 0
                    # self.arg_mapv[p] = []
                    ch_r = self.elems2node[self.clusters[p] - self.clusters[ch_l]]
                    if len(self.clusters[ch_r]) > 1:
                        try:
                            _, r_g, r_h, _, _ = self.peek(ch_r)
                        except IndexError:
                            r_h = self.h_fun(ch_r)
                            r_g = 0
                    else:
                        r_g = r_h = 0
                    g = self.g_fun(ch_l, ch_r) + l_g + r_g
                    h = l_h + r_h
                    f = g + h
                    p_min, _, _, old_l, old_r = self.peek(p)
                    # print('pm, f', p_min, f)
                    # assert p_min <= f
                    self.push(p, (f, g, h, ch_l, ch_r))
                    # print('considering updating parent %s (min val %s) with %s' % (self.clusters[p],
                    #                                                               str(self.pq[p][0]), str((f, g, h,
                    #                                                                                        self.clusters[ch_l],
                    #                                                                                        self.clusters[ch_r]))))
                    # if self.pq[p][0][0] < p_min:
                    if f < p_min:
                        # print('PARENT UPDATE %s: from %s, %s (%s) to %s, %s (%s)' % (self.clusters[p],
                        #                                                              self.clusters[old_l],
                        #                                                              self.clusters[old_r],
                        #                                                              p_min,
                        #                                                              self.clusters[ch_l],
                        #                                                              self.clusters[ch_r],
                        #                                                              f))
                        # raise ValueError
                        nodes_to_update.append(p)
                    # print('<<<<<<<<<<<<<<<<<<<<')
            # print('DONE WITH UPDATES')
            # parent_update = self.update(node)
            # if parent_update:
            #    nodes_to_update.append(parent_update)

        # for i, chs in enumerate(self.children):
        #     if not chs:
        #         continue
        #     for ch in chs:
        #         if ch in hc:
        #             for f, g, h, cl, cr in self.pq[i]:

    def execute_search(self, max_steps=np.Inf, num_matches=0, max_iter=np.Inf, all_pairs_max_size=2, num_tries=4):
        """Runs A* search.

        If num_matches > 0, after A* completes, extend the queues along the
        MAP tree, and then re-runs A* search.
        Repeat this process until the same f value is found num_matches times in a row. This is for the original approx. implementation. Set num_matches=0 for the exact one.

        # Where do we extend the search?

        :param num_matches:
        :return: hc, f - the clustering and f values
        """
        logging.info("Max steps = %s", max_steps)
        hc, f , steps = self._execute_search(max_steps=max_steps, all_pairs_max_size=all_pairs_max_size,num_tries=num_tries )
        logging.info('RESULT -- f: %s -- round %s', f, 1)
        wandb.log({"search_f": f, "search_i": 1})

        ## if num_matches < 0, extend the trellis and then re-run A* until halting condition is met
        ctr = 0
        i = 1
        while ctr < num_matches and i < max_iter:
            logging.debug('matched %s times in a row out of %s times needed', ctr, num_matches)
            self.current_extension_num = i
            # self.reset()
            self.extend_hc_alt()   # This resets values but doesn't extend search
            hc, f_i, steps = self._execute_search(max_steps=max_steps, num_tries=num_tries)
            logging.info('RESULT -- f: %s -- round %s', f_i, i)
            wandb.log({"search_f": f_i, "search_i": i+1})
            if f == f_i:
                ctr += 1
            else:
                ctr = 0
            f = f_i
            i += 1
            # print('after %s times extending and then re-executing search, f value is %s' % (i, f))
            # logging.info('after %s times extending and then re-executing search, f value is $s', i, f)
        ## return the final hc, f
        return hc, f, steps


    # @staticmethod
    def leavesIDX(self, jet, node_id, outers_list):
        """
        Recursive function to get a list of the tree leaves
        """
        if jet["tree"][node_id, 0] == -1:

            outers_list.append(jet["content"][node_id])

        else:
            self.leavesIDX(
            jet,
            jet["tree"][node_id, 0],
            outers_list,
            )

            self.leavesIDX(
            jet,
            jet["tree"][node_id, 1],
            outers_list,
            )

        return outers_list


class IterCCTrellis(IterTrellis):

    def __init__(self, graph: lil_matrix, propagate_values_up: bool, max_nodes: int):
        super(IterCCTrellis, self).__init__(propagate_values_up=propagate_values_up, max_nodes=max_nodes)
        self.graph = graph
        assert self.graph.shape[0] == self.graph.shape[1], "Input graph must be square matrix!"
        # initialize all singleton clusters to be nodes 0 to num points -1
        for i in range(self.graph.shape[0]):
            self.record_node(frozenset({i}))
        self.set_root(frozenset(range(graph.shape[0])))
        logging.info('Root node has elements: %s', self.clusters[self.root])
        logging.info('Root node is id: %s', self.root)

    def g_fun(self, ch_l, ch_r):
        # minimize this: absolute value of the sum of negative edge weights within a cluster
        # plus the sum of positive edge weights across clusters.
        ch_l_ids = np.array(list(self.clusters[ch_l]), dtype=np.int32)
        ch_r_ids = np.array(list(self.clusters[ch_r]), dtype=np.int32)
        assert len(ch_l_ids) > 0
        assert len(ch_r_ids) > 0
        l_within_edges = cartesian_product(ch_l_ids, ch_l_ids)
        r_within_edges = cartesian_product(ch_r_ids, ch_r_ids)
        l_within_edges = l_within_edges[l_within_edges[:, 0] < l_within_edges[:, 1]]
        r_within_edges = r_within_edges[r_within_edges[:, 0] < r_within_edges[:, 1]]
        l_within = self.graph[l_within_edges[:, 0], l_within_edges[:, 1]]
        l_within_neg_sum = l_within[l_within < 0].sum()
        r_within = self.graph[r_within_edges[:, 0], r_within_edges[:, 1]]
        r_within_neg_sum = r_within[r_within < 0].sum()
        across_edges = cartesian_product(ch_l_ids, ch_r_ids)
        across = self.graph[across_edges[:, 0], across_edges[:, 1]]
        pos_across = across[across > 0].sum()
        return pos_across + np.abs(l_within_neg_sum) + np.abs(r_within_neg_sum)

    def h_fun(self, ch_i, sib_i=None):
        return self.sum_of_abs_neg_within(ch_i, sib_i)

    def sum_of_abs_neg_within(self, ch_i, sib_i=None):
        # sum of absolute value of within cluster edge weights
        ch_i_ids = np.array(list(self.clusters[ch_i]), dtype=np.int32)
        assert len(ch_i_ids) > 0
        i_within_edges = cartesian_product(ch_i_ids, ch_i_ids)
        i_within_edges = i_within_edges[i_within_edges[:, 0] < i_within_edges[:, 1]]
        i_within = self.graph[i_within_edges[:, 0], i_within_edges[:, 1]]
        return np.abs(i_within[i_within < 0]).sum()

    def sum_of_abs_within(self, ch_i, sib_i=None):
        # sum of absolute value of within cluster edge weights
        ch_i_ids = np.array(list(self.clusters[ch_i]), dtype=np.int32)
        assert len(ch_i_ids) > 0
        i_within_edges = cartesian_product(ch_i_ids, ch_i_ids)
        i_within_edges = i_within_edges[i_within_edges[:, 0] < i_within_edges[:, 1]]
        i_within = self.graph[i_within_edges[:, 0], i_within_edges[:, 1]]
        return np.abs(i_within).sum()




#########################################################################

class IterJetTrellis(IterTrellis):

    """We want to find a maximum likelihood, so to implement the priority queue with min heap, we flip all the likelihood values. Thus, we will find a solution for the minimum value of (- log LH). Note: we work with (- log LH) for practical purposes """

    def __init__(self, propagate_values_up: bool, max_nodes: int,leaves = None,  min_invM=None, Lambda= None, LambdaRoot=None):
        super(IterJetTrellis, self).__init__(propagate_values_up=propagate_values_up, max_nodes=max_nodes)
        # self.graph = graph
        st = time.time()
        self.leaves_momentum = leaves
        # print("self.leaves_momentum = ", self.leaves_momentum)
        self.momentum = dict()
        self.id_from_momentum= dict()
        self.min_invM = min_invM
        self.Lambda = Lambda.numpy()
        self.LambdaRoot = LambdaRoot.numpy()
        # self.invariant_mass = dict()
        # assert self.graph.shape[0] == self.graph.shape[1], "Input graph must be square matrix!"
        # initialize all singleton clusters to be nodes 0 to num points -1
        # print("self.leaves_momentum.shape[0] = ",self.leaves_momentum.shape[0])
        for i in range(self.leaves_momentum.shape[0]):
            self.momentum[frozenset({i})] = self.leaves_momentum[i]
            self.id_from_momentum[str(self.leaves_momentum[i])] = i
            self.record_node(frozenset({i}))
            # elif len(elements)==1:

        # print("self.momentum=", self.momentum)
        self.set_root(frozenset(range(self.leaves_momentum.shape[0])))
        logging.info('Root node has elements: %s', self.clusters[self.root])
        logging.info('Root node is id: %s', self.root)

        en_t = time.time()- st
        logging.info('Loading class time: %s', en_t)




    # def _sample_children(self, root, all_pairs_max_size=2, num_tries=10):
    #     """Sample 2 partitions up to num_tries"""
    #     # logging.info("Helloooooooo")
    #     cuts = []
    #     root= list(root)
    #     logging.debug("root = %s", root)
    #     # k=0
    #     for _ in range(num_tries):
    #         np.random.shuffle(root)
    #         i = np.random.randint(1,len(root) - 1)
    #         # i += 1
    #         # = frozenset(list(root)[1::])
    #         # cut = (root[:i] , root[i::])
    #         cut = (self.record_node(frozenset(root[0:i])), self.record_node(frozenset(root[i::])))
    #         if cut not in cuts:
    #             cuts += [cut]
    #             # heappush(cuts,cut)
    #             # if len(cuts) == all_pairs_max_size:
    #             #     heappushpop()
    #         # k+=1
    #     # logging.info("k=%s",k)
    #     logging.debug("lenght cuts = %s", len(cuts))
    #     return cuts

    def _sample_children(self, root, all_pairs_max_size=2, num_tries=10, k_max=5):
        """Sample 2 partitions up to num_tries"""
        # logging.info("Helloooooooo")
        cuts = []
        root= list(root)
        logging.debug("root = %s", root)
        # k=0
        for _ in range(num_tries[0]):
            np.random.shuffle(root)
            i = np.random.randint(1,len(root) - 1)
            # i += 1
            # = frozenset(list(root)[1::])
            # cut = (root[:i] , root[i::])
            ch_l= self.record_node(frozenset(root[0:i]))
            ch_r = self.record_node(frozenset(root[i::]))
            g = self.g_fun(ch_l, ch_r)

            cut = (g, ch_l, ch_r)

            if cut not in cuts:
                cuts += [cut]
                # heappush(cuts,cut)
                # if len(cuts) == all_pairs_max_size:
                #     heappushpop()
            # k+=1
        # cuts = sorted(cuts, key=lambda x: x[0])[0:k_max]
        cuts = sorted(cuts, key=lambda x: x[0])[0:num_tries[1]]
        logging.debug("cuts=%s",cuts)
        logging.debug("lenght cuts = %s", len(cuts))
        return [(y,z) for (x,y,z) in cuts]




    def record_node(self, elements: frozenset) -> int:
        """Get the node corresponding to the given elements, create new id if needed.

        Creates a new id if needed.

        :param elements: the elements (FrozenSet of Integers)
        :return: the node for the given elements (this is an index id)
        """
        logging.debug('get node id from elements %s', str(elements))
        if elements not in self.elems2node:
            logging.debug('get node id from elements %s. new node! %s', str(elements), self.next_id)
            self.elems2node[elements] = self.next_id
            self.clusters[self.next_id] = elements
            if len(elements)>1:
                # print('element in elements=', [element for element in elements])
                # print("momentum =", np.asarray([self.momentum[frozenset({elem})] for elem in elements]))
                self.momentum[elements]= sum(np.asarray([self.momentum[frozenset({elem})] for elem in elements])) # Add the momentum of the leaves that compose the node
                # self.invariant_mass[self.next_id] =
            # elif len(elements)==1:
            #     self.momentum[elements]= self.leaves_momentum[list(elements)[0]]

            self.next_id += 1
            return self.next_id - 1
        else:
            return self.elems2node[elements]



    def g_fun(self, ch_l, ch_r):

        return - self.get_energy_of_split(ch_l, ch_r)


    def h_fun(self, ch_i, sib_i=None):

        elements = self.clusters[ch_i]

        # upper_bound = self.upperBoundOuters(elements)
        upper_bound = 0
        if len(elements)>1:
            # print('Is leaf = ', self.is_leaf(ch_i))
            # logging.debug("elements for nodes with more than 1 element = %s", elements)
            # print("self.children[ch_i] =", self.children[ch_i])
            # print("len(self.children[i]) == 0 ? =",len(self.children[ch_i]))
            # print("len(self.pq[i])  == 0?=", len(self.pq[ch_i]))

             # if not self.is_leaf(ch_i):
            # print('Is leaf', self.is_leaf(ch_i))
            # print("elements", elements)
            upper_bound =  self.upperBoundOuters(elements)

            if len(elements)>2:
                upper_bound += self.upperBoundInners(elements)

        return - upper_bound


    def get_energy_of_split(self, ch_l, ch_r):
        """Add last splitting llh to subtree llh.
        Assumes delta_min and lam are the same for both a_node and b_node"""

        logging.debug(f"computing energy of split: {ch_l, ch_r}")

        elem_l = self.clusters[ch_l]
        elem_r = self.clusters[ch_r]

        # To follow the convention on Ginkgo, to get the correct result, we set t==0 if we have a leaf, i.e. t<t_cut
        l_node_invM = 0
        r_node_invM =0
        if len(elem_l)>1: l_node_invM = self.momentum[elem_l][0] ** 2 - np.linalg.norm(self.momentum[elem_l][1::]) ** 2
        if len(elem_r)>1: r_node_invM = self.momentum[elem_r][0] ** 2 - np.linalg.norm(self.momentum[elem_r][1::]) ** 2

        logging.debug(f" t_l ={l_node_invM}")
        logging.debug(f" t_R ={r_node_invM}")

        # logging.debug("elem_l = %s",elem_l)
        logging.debug(f" p_l ={self.momentum[elem_l]}")
        logging.debug(f" p_r ={self.momentum[elem_r]}")

        split_llh = likelihood.split_logLH(self.momentum[elem_l],
                                           l_node_invM,
                                           self.momentum[elem_r],
                                           r_node_invM,
                                           self.min_invM,
                                           self.Lambda)
        # logging.debug(f"split_llh = {split_llh}")

        # llh = split_llh + a_node.map_tree_energy + b_node.map_tree_energy
        logging.debug(f"split likelihood ={split_llh}")

        return split_llh


    def upperBoundOuters(self, elements: frozenset, proof=False):
        """
        We find the minimum possible value for t_p of a leaf. We do this by getting the max t among all leaves and calculate tp_min=t_p and t2=(sqrt(tp)-sqrt(t_max))^2. We could improve this by replacing t_cut by  the 1st t_p that is above t_cut among all pairings of leaves.
        """
        lam = self.Lambda
        t_cut = self.min_invM
        leaves = np.asarray([self.momentum[frozenset({elem})] for elem in elements])
        Nleaves = len(leaves)

        tpi = np.array([
            np.sort([(leaves[k] + leaves[j])[0] ** 2 - np.linalg.norm((leaves[k] + leaves[j])[1::]) ** 2

                     for j in np.concatenate((np.arange(k), np.arange(k + 1, len(leaves))))])
            for k in range(len(leaves))
        ])

        masses_sq = np.sort(
            [leaves[k][0] ** 2 - np.linalg.norm(leaves[k][1::]) ** 2
             for k in range(len(leaves))
             ])

        idxs = [np.searchsorted(entry, t_cut) for entry in tpi]
        #     print(idxs)
        #     print(tpi)
        # t_min = np.sort([tpi[pos, idx] if idx < (Nleaves - 1) else t_cut for pos, idx in enumerate(idxs)])
        # # t_min2 = t_min[np.searchsorted(t_min, t_cut)::]
        # # # logging.info("len t_min=%s | len t_min2=%s", len(t_min), len(t_min2))
        # # # logging.info("diff len t_min -len t_min2=%s", len(t_min)-len(t_min2))
        # # if len(t_min) > len(t_min2):
        # #     t_min2 = np.concatenate((t_min2, t_min2[-1] * (len(t_min) - len(t_min2))))
        #
        # # logging.info("t_min2[0]= %s", t_min2[0])
        #
        # tp2 = (np.sqrt(t_min[0]) - np.sqrt(masses_sq)) ** 2

        # t_min = np.asarray([tpi[pos, idx] if idx < (Nleaves - 1) else t_cut for pos, idx in enumerate(idxs)])
        #
        # #     t_min2= t_min[np.searchsorted(t_min, t_cut) ::]
        # #     print(t_min2)
        # #     if len(t_min)>len(t_min2):
        # #         t_min2 =np.concatenate((t_min2, t_min2[-1]*(len(t_min)-len(t_min2))))
        #
        # tp2 = (np.sqrt(t_min) - np.sqrt(masses_sq[-1])) ** 2
        # tp2[-1] = t_min[-1]

        t_min = np.sort([tpi[pos, idx] if idx < (Nleaves - 1) else t_cut for pos, idx in enumerate(idxs)])


        if proof:
            """Exact  result with proof"""
            tp2 = (np.sqrt(t_min) - np.sqrt(masses_sq[-1])) ** 2
        else:
            """Exact result with no proof"""
            t_min2 = t_min[np.searchsorted(t_min, t_cut)::]
            tp2 = (np.sqrt(t_min2[0]) - np.sqrt(masses_sq)) ** 2
            tp2[-1] = t_min2[0]



        #     tmax=    max( [ leaves[k][0]**2-np.linalg.norm(leaves[k][1::])**2
        #            for k in range(len(leaves))
        #           ])
        #     tp2 = (np.sqrt(t_cut) - np.sqrt(tmax)) ** 2

        llh = -np.log(1 - np.exp(- lam)) + np.log(1 - np.exp(-lam * t_cut / tp2))

        return sum(llh)


    # def upperBoundOuters(self, elements: frozenset):
    #     """
    #     We find the minimum possible value for t_p of a leaf. We do this by getting the max t among all leaves and calculate tp_min=t_p and t2=(sqrt(tp)-sqrt(t_max))^2. We could improve this by replacing t_cut by  the 1st t_p that is above t_cut among all pairings of leaves.
    #     """
    #     lam = self.Lambda
    #     t_cut = self.min_invM
    #     leaves = np.asarray([self.momentum[frozenset({elem})] for elem in elements])
    #     Nleaves = len(leaves)
    #
    #     tmax = max([leaves[k][0] ** 2 - np.linalg.norm(leaves[k][1::]) ** 2
    #                 for k in range(len(leaves))
    #                 ])
    #     tp2 = (np.sqrt(t_cut) - np.sqrt(tmax)) ** 2
    #
    #     llh = -np.log(1 - np.exp(- lam)) + np.log(1 - np.exp(-lam * t_cut / tp2))
    #
    #     return llh * Nleaves



    def upperBoundInners(self, elements: frozenset, proof=False):
        """ Get upper bound on logLH of all inner nodes.
        We bound t_p in the exponential by "jet mass" which is the invariant mass of the sum of the leaves under each subtree.
        We bound t_p in the denominator and t in the exponential by the max{min tp among all leaves pairings, t_cut}. A better bound but more time consuming would be to get the 1st t_p that is above t_cut. These are inner nodes, so the minimum possible value of t_p (their parent) is given by the minimum pairing of leaves above t_cut, regardless of the subtlety of t1=t_p and t2=(sqrt(tp)-sqrt(t_max))^2. Because in the worst case scenario t_p=t and we have t>t_cut.
        """
        lam = self.Lambda
        lamRoot = self.LambdaRoot
        t_cut = self.min_invM
        leaves = np.asarray([self.momentum[frozenset({elem})] for elem in elements])
        # print("Leaves in current bound = ", leaves)
        # print("All leaves = ", self.leaves_momentum )
        # print("Leaves 0in bound = ", leaves[0])
        Nleaves = len(leaves)

        subjet_vec = sum(leaves)
        subjetMass2 = subjet_vec[0] ** 2 - np.linalg.norm(subjet_vec[1::]) ** 2
        #     print("subjetMass2 =", subjetMass2)

        """Find minimum invariant mass for each leaf possible parent, i.e. for each leaf find the pair that minimizes the invariant mass"""
        tpi = np.array([
            np.sort([(leaves[k] + leaves[j])[0] ** 2 - np.linalg.norm((leaves[k] + leaves[j])[1::]) ** 2

                     for j in np.concatenate((np.arange(k), np.arange(k + 1, len(leaves))))])
            for k in range(len(leaves))
        ])

        masses_sq = np.sort(
            [leaves[k][0] ** 2 - np.linalg.norm(leaves[k][1::]) ** 2
             for k in range(len(leaves))
             ])

        idxs = [np.searchsorted(entry, t_cut) for entry in tpi]



        t_min = np.sort([tpi[pos, idx] if idx < (Nleaves - 1) else t_cut for pos, idx in enumerate(idxs)])
        t_min2 = t_min[np.searchsorted(t_min, t_cut)::]

        values = t_min2[0:Nleaves // 2]
        i = 3
        j = Nleaves % 2 + Nleaves // 2
        #     print('values=',values)

        while len(values) < Nleaves - 2:
            #         values = np.concatenate((values, [max((masses_sq[0] *(i%2)+ t_min2[0]*(i//2)) * (j// 2), sum(masses_sq[0:i]))] * (j // 2)))
            values = np.concatenate((values, [(masses_sq[0] * (i % 2) + t_min2[0] * (i // 2)) * (j // 2)]))
            j = j // 2 + j % 2
            i += 1


        #     print(idxs)
        #     print(tpi)
        # t_min2 = np.sort([tpi[pos, idx] if idx < (Nleaves - 1) else t_cut for pos, idx in enumerate(idxs)])
        # # t_min2 = t_min[np.searchsorted(t_min, t_cut)::]
        # # #     print(t_min2)
        # # if len(t_min) > len(t_min2):
        # #     t_min2 = np.concatenate((t_min2, t_min2[-1] * (len(t_min) - len(t_min2))))
        # #     print(t_min)
        #
        # #     values=[]
        # #     print(np.concatenate(np.empty(),t_min[0:Nleaves//2]))
        # values = t_min2[0:Nleaves // 2]
        # i = 3
        # j = Nleaves % 2 + Nleaves // 2
        # #     print('values=',values)
        #
        # while len(values) < Nleaves - 2:
        #     values = np.concatenate((values, [max(t_cut + t_min2[0] * (j - 2), sum(masses_sq[0:i]))] * (j // 2)))
        #     j = j // 2 + j % 2
        #     i += 1

        # These are inner nodes so the 1st tp is for 3 elements
        #     tp_Max = t_min2[-1:]
        #     print(tp_Max)

        #     t_pairMax = np.max([
        #                 np.max([   (leaves[k]+leaves[j])[0]**2-np.linalg.norm((leaves[k]+leaves[j])[1::])**2

        #                 for j in np.concatenate((np.arange(k),np.arange(k+1,len(leaves)))) ])
        #            for k in range(len(leaves))
        #           ])

        # k = 2
        # #     t_prev = 3* t_pairMax+masses_sq[-(k+1)]
        # t_prev = 3 * (3 * masses_sq[-(k - 1)] + masses_sq[-(k)]) + masses_sq[-(k + 1)]
        # tp_Max = [min(t_prev, subjetMass2)]
        # while len(tp_Max) < Nleaves - 2:
        #     k += 1
        #     t_prev = 3 * t_prev + masses_sq[-(k + 1)]
        #     tp_Max = np.concatenate((tp_Max, [min(t_prev, subjetMass2)]))


        #         print('values=',values)
        #         print(j)
        #         print(i)
        #     values=np.asarray(values)

        #     print('tp_Max= ',tp_Max)
        #     print('jetMass2 = ',jetMass2)
        #     print('---------------')

        # if len(values) != len(tp_Max):
        #     print('Nleaves = ', Nleaves)
        #     print('len(values) =', len(values))
        #     print(len(tp_Max))

        if proof:
            """Exact  result with proof"""
            values_p = [entry + masses_sq[0] if entry > masses_sq[-1] else entry for entry in values]
        else:
            """Exact  result with no proof"""
            values_p = values + 2* t_min2[0]

        #     values_p= [(values[i]+masses_sq[i]) if values[i]>masses_sq[-1] else values[i] for i in range(len(values))]
        #     llh = -np.log(1 - np.exp(- lam)) + np.log(lam) - np.log(values+t_min2[0]) - lam * values / tp_Max
        llh = -np.log(1 - np.exp(- lam)) + np.log(lam) - np.log(values_p) - lam * values / subjetMass2

        # llh = -np.log(1 - np.exp(- lam)) + np.log(lam) - np.log(values + t_min2[0]) - lam * values / tp_Max
        # llh = -np.log(1 - np.exp(- lam)) + np.log(lam) - np.log(values + t_min2[0]) - lam * values / subjetMass2

        return sum(llh[0:len(leaves) - 2]) + (len(leaves) - 1) * np.log(1 / (4 * np.pi))


        # def upperBoundInners(self, elements: frozenset):
    #     """ Get upper bound on logLH of all inner nodes.
    #     We bound t_p in the exponential by "jet mass" which is the invariant mass of the sum of the leaves under each subtree.
    #     We bound t_p in the denominator and t in the exponential by the max{min tp among all leaves pairings, t_cut}. A better bound but more time consuming would be to get the 1st t_p that is above t_cut. These are inner nodes, so the minimum possible value of t_p (their parent) is given by the minimum pairing of leaves above t_cut, regardless of the subtlety of t1=t_p and t2=(sqrt(tp)-sqrt(t_max))^2. Because in the worst case scenario t_p=t and we have t>t_cut.
    #     """
    #     lam = self.Lambda
    #     lamRoot = self.LambdaRoot
    #     t_cut = self.min_invM
    #     leaves = np.asarray([self.momentum[frozenset({elem})] for elem in elements])
    #     # print("Leaves in current bound = ", leaves)
    #     # print("All leaves = ", self.leaves_momentum )
    #     # print("Leaves 0in bound = ", leaves[0])
    #     Nleaves = len(leaves)
    #
    #     subjet_vec = sum(leaves)
    #     subjetMass2 = subjet_vec[0] ** 2 - np.linalg.norm(subjet_vec[1::]) ** 2
    #     #     print("subjetMass2 =", subjetMass2)
    #
    #     """Find minimum invariant mass for each leaf possible parent, i.e. for each leaf find the pair that minimizes the invariant mass"""
    #     # tpi = [
    #     #     min([(leaves[k] + leaves[j])[0] ** 2 - np.linalg.norm((leaves[k] + leaves[j])[1::]) ** 2
    #     #          for j in np.concatenate((np.arange(k), np.arange(k + 1, len(leaves))))])
    #     #     for k in range(len(leaves))
    #     # ]
    #
    #     # tpi = np.sort(tpi)
    #     # # #    Find the smallest tpi that is greater than t_cut
    #     # idx = np.searchsorted(tpi, t_cut)
    #     # if idx >= len(tpi) - 1:
    #     #     t_min = t_cut
    #     # else:
    #     #     t_min = tpi[idx]
    #     #
    #     #
    #     # llh = -np.log(1 - np.exp(- lam)) + np.log(lam) - np.log(t_min) - lam * t_min / subjetMass2
    #     #
    #     # # if elements == self.clusters[self.root]:
    #     # #     llhRoot = -np.log(1 - np.exp(- lamRoot)) + np.log(lamRoot) - np.log(subjetMass2) - lamRoot * t_min / subjetMass2
    #     #
    #     # # return llhRoot + (len(leaves) - 2) * llh + (Nleaves - 1) * np.log(1 / (4 * np.pi))
    #     # return  (len(leaves) - 2) * llh + (len(leaves) - 1) * np.log(1 / (4 * np.pi))
    #     #     # (len(leaves) - 2) * ( llh + np.log(1 / (4 * np.pi)) )
    #
    #     tpi = np.array([
    #         np.sort([(leaves[k] + leaves[j])[0] ** 2 - np.linalg.norm((leaves[k] + leaves[j])[1::]) ** 2
    #
    #                  for j in np.concatenate((np.arange(k), np.arange(k + 1, len(leaves))))])
    #         for k in range(len(leaves))
    #     ])
    #
    #     idxs = [np.searchsorted(entry, t_cut) for entry in tpi]
    #     #     print(idxs)
    #     #     print(tpi)
    #     t_min = np.sort([tpi[pos, idx] if idx < (Nleaves - 1) else t_cut for pos, idx in enumerate(idxs)])
    #     #     print(t_min)
    #
    #     llh = -np.log(1 - np.exp(- lam)) + np.log(lam) - np.log(t_min) - lam * t_min / subjetMass2
    #
    #     return sum(llh[0:len(leaves) - 2]) + (len(leaves) - 1) * np.log(1 / (4 * np.pi))