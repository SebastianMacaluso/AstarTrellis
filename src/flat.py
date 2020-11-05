
from trellis.iter_trellis import IterTrellis
import numpy as np

from absl import logging
from heapq import heappush
from itertools import combinations

def all_two_partitions_flat(elem):
    if len(elem) == 1:
        return [(elem + [-1], None)]
    assert len(elem) > 1
    special_elem, other_elems = elem[0], elem[1:]
    cuts = [([special_elem], other_elems)]
    for i in range(1, len(other_elems)):
        for el in combinations(other_elems, i):
            l_child = [special_elem] + list(el)
            r_child = [i for i in elem if not i in l_child]
            cuts.append((l_child, r_child))
    cuts.append((elem + [-1], None))
    logging.debug('all_two_partitions_flat of %s gives %s values', str(elem), len(cuts))
    return cuts

class FlatTrellis(IterTrellis):
    """

    Extends the trellis by having "leaf" nodes
    that correspond to clusters rather than singletons
    as the leaves.

    We implement this by adding one more pair of children for each node.
    That correspond to the node itself and the value None.

    """
    def __init__(self, max_nodes):
        super(FlatTrellis, self).__init__(max_nodes, False)

    def get_state(self):
        """Return the best available (partial) hierarchical clustering.

        :return: in_flat_partition - the flat partition found
        :return: internals - internal nodes
        :return: lvs - leaf nodes
        :return: parent2children - parent2children map
        """
        logging.debug('Getting current state.')
        travel = [self.root]
        hc = []
        internals = []
        lvs = []
        parent2children = dict()
        in_flat_partition = []
        while travel:
            i = travel.pop()
            hc.append(i)
            logging.debug('visiting node %s', i)
            if self.is_leaf(i):
                logging.debug('visiting node %s - is leaf', i)
                lvs.append(i)
            else:
                logging.debug('visiting node %s - is iternal', i)
                internals.append(i)
            if self.explored[i]:
                a, b = self.walk_down(i)
                logging.debug('visiting node %s - has kids %s and %s', i, a, b)
                parent2children[i] = [a, b]
                self.memoize_if_needed(i)
                if a is not None and b is not None:
                    travel.append(a)
                    travel.append(b)
                if a is None:
                    assert b is not None
                    in_flat_partition.append(b)
                    lvs.append(b)
                if b is None:
                    assert a is not None
                    in_flat_partition.append(a)
                    lvs.append(a)

        return in_flat_partition, internals, lvs, parent2children

    def is_goal_state(self, hc, internals, lvs):
        """Return true if this is a goal state.

        if leaves is 0 then everything must be in the flat partition!

        :param hc: the full tree
        :param internals: the internal nodes
        :param lvs: the leaves
        :return: true if this is a goal state.
        """
        res = all([-1 in self.clusters[x] for x in lvs]) and len(hc) > 0
        return res

    def extend(self, i, elems):
        """Extend the node i.

        This extends the queue the node i.
        It sets the priority queue of i accordingly.

        :param i: the node i
        :param elems: the elements at the node
        :return:
        """
        assert -1 not in elems, 'Special leaf token found in node being extended.'
        ch = self._get_children(list(elems))
        logging.debug('extend node %s with %s children', i, len(ch))

        for l, r in ch:
            assert l is not None or r is not None
            ch_l = self.record_node(frozenset(l)) if l else None
            ch_r = self.record_node(frozenset(r)) if r else None
            if (ch_l, ch_r) in self.children[i]:
                continue
            self.children[i].append((ch_l, ch_r))
            g = self.g_fun(ch_l, ch_r)
            h = 0
            logging.debug('extending node %s with children %s and %s | g %s | h %s', i, ch_l, ch_r, g, h)
            for ch_i, sib_i in [(ch_l, ch_r), (ch_r, ch_l)]:
                if ch_i is not None:
                    # if the child is already there, grab values from it
                    if self.explored[ch_i]:
                        _, g_i, h_i, _, _ = self.pq[ch_i][0]
                        g += g_i
                        h += h_i
                    else:
                        h += self.h_fun(ch_i, sib_i=sib_i)
                    self.push(i, (g + h, g, h, ch_l, ch_r))
        self.explored[i] = True
        return ch

    def update_from_children(self, p, children):
        """Update the parents priority queue.

        :param p: the parent id
        :param children: list (of size 2) of the children ids
        :return:
        """
        logging.debug('update %s from the children %s', p, str(children))
        ch_l, ch_r = children
        g = self.g_fun(ch_l, ch_r)
        h = 0
        for ch_i, sib_i in [(ch_l, ch_r), (ch_r, ch_l)]:
            if ch_i is not None:
                # if the child is already there, grab values from it
                if self.explored[ch_i]:
                    _, g_i, h_i, _, _ = self.pq[ch_i][0]
                    g += g_i
                    h += h_i
                else:
                    h += self.h_fun(ch_i, sib_i=sib_i)
        self.pop(p)
        self.push(p, (g + h, g, h, ch_l, ch_r))

class DPMeansTrellis(FlatTrellis):

    def __init__(self, points, opening_cost, max_nodes: int):
        super(DPMeansTrellis, self).__init__(max_nodes)
        self.points = points
        # initialize all singleton clusters to be nodes 0 to num points -1
        for i in range(self.points.shape[0]):
            self.record_node(frozenset({i}))
        self.set_root(frozenset(range(points.shape[0])))
        logging.info('Root node has elements: %s', self.clusters[self.root])
        logging.info('Root node is id: %s', self.root)
        self.opening_cost = opening_cost

    def g_fun(self, ch_l, ch_r):
        # minimize this: absolute value of the sum of negative edge weights within a cluster
        # plus the sum of positive edge weights across clusters.
        if ch_l is not None and ch_r is not None:
            return 0
        else:
            ch_i = ch_l if ch_l is not None else ch_r
            ch_i_ids = np.array([x for x in list(self.clusters[ch_i]) if x != -1], dtype=np.int32)
            assert len(ch_i_ids) > 0
            this_cluster_points = self.points[ch_i_ids]
            center = np.mean(this_cluster_points, axis=0, keepdims=True)
            from scipy.spatial.distance import cdist
            dists = cdist(center, this_cluster_points, metric='sqeuclidean')
            return np.sum(dists) + self.opening_cost

    def h_fun(self, ch_i, sib_i=None):
        if sib_i is None:
            return 0
        else:
            assert ch_i is not None and ch_i is not None
            ch_i_ids = np.array([x for x in list(self.clusters[ch_i]) if x != -1], dtype=np.int32)
            assert len(ch_i_ids) > 0
            this_cluster_points = self.points[ch_i_ids]
            from scipy.spatial.distance import cdist
            dists = cdist(this_cluster_points, this_cluster_points, metric='sqeuclidean')
            dists *= 0.5
            np.fill_diagonal(dists, np.Inf)
            lower_bound = np.sum(np.minimum(np.min(dists, axis=1), self.opening_cost))
            return lower_bound

    def is_leaf(self, i):
        """ Returns true if i is a leaf.

        True if i has no children

        :param i: node id
        :return: if there are no children.
        """
        return len(self.children[i]) == 0 or len(self.pq[i]) == 0

    def get_children(self, i, elems):
        """Gives the children of node i that has elements elems.

        In this version, it grabs all 2 partitions if they are not there
        and caches this in children[i].

        :param i: The node i
        :param elems: The elements associated with node i
        :return: the children of i
        """
        if -1 in elems:
            return []
        elif self.children[i]:
            return self.children[i]
        else:
            chp = self._get_children(list(elems))
            ch = [(self.record_node(frozenset(l)) if l else None, self.record_node(frozenset(r)) if r else None) for l, r in chp]
            self.children[i] = ch
            return ch

    def initialize(self, i, elems):
        """Initialize the node i.

        This explores the node i. It sets the priority queue of i accordingly.

        :param i: the node i
        :param elems: the elements at the node
        :return:
        """
        ch = self.get_children(i, elems)
        logging.debug('initialize node %s with %s children', i, len(ch))

        for ch_l, ch_r in ch:
            g = self.g_fun(ch_l, ch_r)
            h = 0
            logging.debug('initialize node %s with children %s and %s | g %s | h %s', i, ch_l, ch_r, g, h)
            for ch_i, sib_i in [(ch_l, ch_r), (ch_r, ch_l)]:
                if ch_i is None:
                    continue
                # if the child is already there, grab values from it
                if self.explored[ch_i]:
                    _, g_i, h_i, _, _ = self.pq[ch_i][0]
                    g += g_i
                    h += h_i
                else:
                    h += self.h_fun(ch_i, sib_i=sib_i)
                logging.debug('initialize node %s with children %s and %s | g %s | h %s', i, ch_l, ch_r, g, h)

            self.explored[i] = True
            self.push(i, (g + h, g, h, ch_l, ch_r))

    def memoize_if_needed(self, i):
        """Memoize the values of i if we know them.

        :param i: node id
        :return: Nothing.
        """
        f, g, h, a, b = self.pq[i][0]
        assert a is not None or b is not None
        if h == 0.0 and (a is None or self.last_memoized[a] == self.current_extension_num) \
          and (b is None or self.last_memoized[b] == self.current_extension_num):
             logging.debug('memoizing node %s - g value %s - children %s and %s', i, g, a, b)
             self.mapv[i] = g
             self.arg_mapv[i] = [a, b]
             self.last_memoized[i] = self.current_extension_num

    def load_from_tree(self, tree, hc):
        ## remove existing parameters and then replace them from tree. not LEAVE IN PLACE graph and _get_children
        FlatTrellis.__init__(self, max_nodes=len(self.children))

        self.next_id = tree.next_node_id

        # set children
        for i, ch in enumerate(tree.children):
            if i > tree.next_node_id:
                break
            if not ch:
                continue
            self.children[i].append(tuple(ch))

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

        # add self-child
        res = dict()
        for elems, node_id in self.elems2node.items():
            if -1 not in elems:
                new_node_id = self.next_id
                self.next_id += 1
                ch_elm = frozenset(list(elems) + [-1])
                self.clusters[new_node_id] = ch_elm
                self.children[node_id].append((new_node_id, None))
                res[ch_elm] = new_node_id
        self.elems2node.update(res)

        ## sort nodes from small to large (based on # of elements)
        ## for each node:
        # ## find node's children
        # ## get g's from children and psi function, set h = 0
        # ## set pq, mapv, argmapv
        # for elems in sorted(self.elems2node, key=lambda k: len(k)):
        #     node = self.elems2node[elems]
        #     # print(elems, node)
        #     if -1 in elems:
        #         # self.mapv[node] = 0
        #         # self.arg_mapv[node] = node
        #         continue
        #     # print('elems', elems)
        #     # print('node', node)
        #     # print('children', self.children[node])
        #     self.explored[node] = True
        #     ch_l, ch_r = self.children[node][0]
        #     g = self.g_fun(ch_l, ch_r)
        #     h = self.h_fun(ch_l,ch_r) + self.h_fun(ch_r,ch_l)
        #     self.arg_mapv[node] = [ch_l, ch_r]
        #     # heappush(self.pq[node], (self.mapv[node], self.mapv[node], 0, ch_l, ch_r))
        #     self.push(node, (g+h, g, 0, ch_l, ch_r))
