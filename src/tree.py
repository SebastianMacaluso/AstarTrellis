
import pickle
import time

import higra as hg
import numpy as np
from absl import app
from absl import logging
from scipy.sparse import load_npz,coo_matrix

def sparse_avg_hac(coo_pw_sim_mat, convert_to_dense_and_pad=True):
    """Run hac on a coo sparse matrix of edges.


    :param coo_pw_sim_mat: N by N coo matrix w/ pairwise sim matrix
    :return: Z - linkage matrix, as in scipy linkage, other meta data from higra
    """
    if convert_to_dense_and_pad:
        d = coo_pw_sim_mat.todense()
        d += 1e-12
        coo_pw_sim_mat = coo_matrix(d)
    ugraph, edge_weights = coo_2_hg(coo_pw_sim_mat)
    t, altitudes = hg.binary_partition_tree_average_linkage(ugraph, edge_weights)
    Z = hg.binary_hierarchy_to_scipy_linkage_matrix(t,altitudes=altitudes)
    return Z, t, altitudes, ugraph, edge_weights

def coo_2_hg(coo_mat):
    """Convert coo matrix to higra input format."""
    rows = coo_mat.row[coo_mat.row < coo_mat.col]
    cols = coo_mat.col[coo_mat.row < coo_mat.col]
    sims = coo_mat.data[coo_mat.row < coo_mat.col]
    dists = sims.max() - sims
    ugraph = hg.higram.UndirectedGraph(coo_mat.shape[0])
    ugraph.add_edges(rows.tolist(),cols.tolist())
    return ugraph, dists.astype(np.float32)

def load_graph(filename):
    init_knn_dist = load_npz(filename).tocoo()
    return init_knn_dist


class Tree(object):

    def __init__(self, graph):
        num_points = graph.shape[0]
        self.graph = graph.tocoo()
        self.num_points = num_points
        self.max_nodes = 100000
        self.pad_idx = self.max_nodes-1
        self.ancs = [[] for _ in range(self.max_nodes)]
        self.children = [[] for _ in range(self.max_nodes)]
        self.descendants = [[] for _ in range(self.max_nodes)]
        for i in range(self.num_points):
            self.descendants[i] = [i]
        self.scores = -np.inf*np.ones(self.max_nodes,dtype=np.float32)
        self.parent = -1*np.ones(self.max_nodes, dtype=np.int32)
        self.next_node_id = num_points
        self.num_descendants = -1 * np.ones(self.max_nodes, dtype=np.int32)
        self.needs_update_desc = np.ones(self.max_nodes, dtype=np.bool_)
        self.needs_update_desc[:self.num_points] = 0
        #self.next_node_id = num_points ## set above

    def execute_search(self):
        Zavg, tree, alt, ugraph, edge_weights = sparse_avg_hac(self.graph)
        self.from_scipy_z(Zavg)
        c = self.cc_cost(self.graph.tolil())
        return self, c

    def agglom(self, edge_coo: coo_matrix):

        frontier = []
        from heapq import heappush, heappop
        for idx in range(edge_coo.shape[0]):
            i = edge_coo.row[idx]
            j = edge_coo.col[idx]
            c = edge_coo.data[idx]
            wi_p = c if c > 0 else 0.0
            wi_n = c if c < 0 else 0.0
            ac_p = 0.0
            ac_n = 0.0
            rev = np.abs(wi_p)
            heappush(frontier, (-rev, wi_p, wi_n, ac_p, ac_n, i, j))
        roots = set([x for x in range(self.num_points)])
        def valid_move(i, j):
            return i in roots and j in roots
        edge_lil = edge_coo.tolil()
        def new_nodes(nni, roots):
            nn = []
            for r in roots:
                wi_p = 0.0
                wi_n = 0.0
                ac_p = 0.0
                ac_n = 0.0
                for i in self.descendants[nni]:
                    for k in self.descendants[r]:
                        if edge_lil[i, k] > 0.0:
                            wi_p += edge_lil[i, k]
                rev = wi_p
                nn.append((-rev, wi_p, wi_n, ac_p, ac_n, nni, r))
            return nn

        while frontier:
            rev, wi_p, wi_n, ac_p, ac_n, i, j = heappop(frontier)
            logging.debug('valid_move(%s, %s) = %s | rev=%s', i, j, valid_move(i, j), rev)

            while not valid_move(i, j) and frontier:
                logging.debug('valid_move(%s, %s) = %s | rev=%s', i, j, valid_move(i, j), rev)
                rev, wi_p, wi_n, ac_p, ac_n, i, j = heappop(frontier)
            if not valid_move(i, j):
                logging.debug('no valid move, no frontier, we are done')
                assert len(roots) == 1
                return
            nni = self.next_node_id
            logging.debug('Merging %s and %s into %s with rev=%s | %s roots', i, j, nni, rev, len(roots))
            self.next_node_id += 1
            self.descendants[nni].extend(self.descendants[i])
            self.descendants[nni].extend(self.descendants[j])
            self.children[nni] = [i, j]
            self.parent[i] = nni
            self.parent[j] = nni
            roots.remove(i)
            roots.remove(j)
            for r in new_nodes(nni, roots):
                heappush(frontier, r)
            roots.add(nni)

    def from_scipy_z(self, Z):
        assert self.num_points == Z.shape[0] + 1
        for i in range(Z.shape[0]):
            internal_id = i + self.num_points
            self.parent[Z[i, 0].astype(np.int32)] = internal_id
            self.parent[Z[i, 1].astype(np.int32)] = internal_id
            self.children[internal_id] = [Z[i, 0].astype(np.int32), Z[i, 1].astype(np.int32)]

        self.update_desc(self.root())
        self.next_node_id = self.num_points*2

    def root(self, r=0):
        while self.get_parent(r) != -1:
            r = self.get_parent(r)
        return r

    def update_desc(self, i):
        s = time.time()
        needs_update = []
        to_check = [i]
        while to_check:
            curr = to_check.pop(0)
            if self.needs_update_desc[curr]:
                needs_update.append(curr)
                for c in self.get_children(curr):
                    to_check.append(c)
        for j in range(len(needs_update)-1,-1,-1):
            self.single_update_desc(needs_update[j])

    def single_update_desc(self, i):
        assert self.needs_update_desc[i]
        kids = self.get_children(i)
        self.descendants[i].clear()
        self.descendants[i].extend(self.descendants[kids[0]])
        self.descendants[i].extend(self.descendants[kids[1]])
        self.num_descendants[i] = len(self.descendants[i])
        self.needs_update_desc[i] = False

    def get_children(self, i):
        return self.children[i]

    def get_parent(self, i):
        return self.parent[i]

    # https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
    def cartesian_product(self, *arrays):
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        return arr.reshape(-1, la)

    def dasgupta_cost(self, graph):
        internal_nodes = [i for i in range(self.next_node_id) if self.children[i]]
        overall_cost = 0.0
        for n in internal_nodes:
            c1 = self.children[n][0]
            c2 = self.children[n][1]
            c1_desc = np.array(self.descendants[c1], dtype=np.int32)
            c2_desc = np.array(self.descendants[c2], dtype=np.int32)
            pairs = self.cartesian_product(c1_desc, c2_desc)
            # pairs = pairs[pairs[:,0] < pairs[:,1], :]
            cut_cost = graph[pairs[:, 0], pairs[:, 1]].sum()
            self.scores[n] = cut_cost
            overall_cost += self.num_descendants[n] * cut_cost
            # print(pairs, cut_cost, self.num_descendants[n])
        return overall_cost

    def cc_cost(self, graph):
        internal_nodes = [i for i in range(self.next_node_id) if self.children[i]]
        assert self.root() in internal_nodes
        overall_cost = 0.0
        for n in internal_nodes:
            c1 = self.children[n][0]
            c2 = self.children[n][1]
            c1_desc = np.array(self.descendants[c1], dtype=np.int32)
            c2_desc = np.array(self.descendants[c2], dtype=np.int32)
            pairs = self.cartesian_product(c1_desc, c2_desc)
            within_pos = 0.0
            for d1 in c1_desc:
                for d2 in c1_desc:
                    if graph[d1, d2] < 0 and d1 < d2:
                        within_pos += np.abs(graph[d1, d2])

            for d1 in c2_desc:
                for d2 in c2_desc:
                    if graph[d1, d2] < 0 and d1 < d2:
                        within_pos += np.abs(graph[d1, d2])
            ac = graph[pairs[:, 0], pairs[:, 1]]

            cost = within_pos + np.abs(ac[ac > 0]).sum()
            self.scores[n] = cost
            overall_cost += cost
            # print(pairs, cut_cost, self.num_descendants[n])
        return overall_cost

    def from_set_of_sets(self, set_of_sets):
        s2id = dict()
        singletons = [x for x in set_of_sets if len(x) == 1]
        singletons = sorted(singletons,key=lambda x: next(iter(next(iter(x)))))
        for idx,s in enumerate(singletons):
            s2id[s] = idx
        nextid = len(singletons)
        print(s2id)

        def subset(x, y):
            return len(x) == len(x.intersection(y)) and x != y

        for s in set_of_sets:
            candidate_parents = [x for x in set_of_sets if subset(s, x)]
            if s not in s2id:
                s2id[s] = nextid
                nextid += 1
            if candidate_parents:
                p = min(candidate_parents, key=lambda x: len(x))

                if p not in s2id:
                    s2id[p] = nextid
                    nextid += 1
                self.parent[s2id[s]] = s2id[p]
                self.children[s2id[p]].append(s2id[s])

        self.update_desc(self.root())
        self.next_node_id = nextid


def check_bidrection(graph: coo_matrix):
    rows = graph.row
    cols = graph.col
    lil = graph.tolil()
    for i in range(rows.shape[0]):
        assert np.abs(lil[rows[i], cols[i]] - lil[cols[i], rows[i]]) < 1e-5, "%s -> %s = %s, %s -> %s = %s" % (rows[i], cols[i], lil[rows[i], cols[i]], cols[i], rows[i], lil[cols[i], rows[i]])

def main(argv):
    logging.info('Running with args %s', str(argv))
    graphs = pickle.load(open('/tmp/cc_aloi_samples.pt10.k20.pkl', 'rb'))
    costs = []
    for grapha in graphs:
        graph = grapha[0]
        graph.data = graph.data

        # print(c)
        # print(np.unique(grapha[1][1]))
        # idmap = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7:'h', 8:'i', 9:'j'}
        # res = []
        # for idx,y in enumerate(t.descendants[:21]):
        #     print('%s (%s)' % (str([idmap[x] for x in y]),t.scores[idx]))
        #     res.append([idmap[x] for x in y])
        # print('\n'.join([str(x) for x in res]))
    print(costs)
    print(np.mean(costs))
    print(np.std(costs))
    print()
    print('\n')
    print('\n'.join([str(x) for x in costs]))

if __name__ == "__main__":
    app.run(main)
    # eval_craig_tree2()


