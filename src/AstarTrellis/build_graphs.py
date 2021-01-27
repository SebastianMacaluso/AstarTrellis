import os
import numpy as np
import time
import torch
import scipy.sparse
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix


try:
    from absl import logging
    from absl import flags
    from absl import app
except ImportError:
    raise Exception('Please install absl-py: pip install absl-py')

logging.set_verbosity(logging.INFO)

def load_text(infile, unit_norm):
    vecs = []
    lbls = []
    with open(infile, 'r') as f:
        for i, line in enumerate(f):
            splits = line.strip().split('\t')
            lbls.append(splits[1])
            vecs.append([float(x) for x in splits[2:]])
    vecs = np.array(vecs, dtype=np.float32)
    if unit_norm == 1:
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return np.arange(vecs.shape[0]), lbls, vecs


def load_xcluster(filename, normalize):
    data = np.loadtxt(filename)
    point_ids = data[:, 0].astype(np.int32)
    point_labels = data[:, 1].astype(np.int32)
    vectors = data[:, 2:].astype(np.float64)
    if normalize == 1:
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return point_ids, point_labels, vectors

def load_npy_matrix(filename, normalize):
    vectors = np.load(filename).astype(np.float64)
    point_ids = np.arange(vectors.shape[0], np.int32)
    point_labels = np.arange(vectors.shape[0], np.int32)
    if normalize == 1:
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return point_ids, point_labels, vectors

def load(filename, format, normalize):
    if format == 'xcluster':
        return load_xcluster(filename, normalize)
    elif format == 'npy':
        return load_npy_matrix(filename, normalize)
    elif format == 'text':
        return load_text(filename, normalize)
    else:
        raise Exception('Unknown input format %s' % format)

def knn(query_vectors, base_vectors, batch_size=3, K=50):
    t = time.time()
    distances = np.zeros((query_vectors.shape[0], K), dtype=np.float32)
    indices = np.zeros((query_vectors.shape[0], K), dtype=np.int32)
    for i in range(0, query_vectors.shape[0], batch_size):
        topk = torch.topk(torch.matmul(query_vectors[i:(i+batch_size)], base_vectors), k=K, dim=1)
        d, idx = topk[0].cpu().numpy(), topk[1].cpu().numpy()
        distances[np.arange(i, i + d.shape[0])] = d.astype(np.float32)
        indices[np.arange(i, i + d.shape[0])] = idx.astype(np.int32)
        logging.info('Finished %s out of %s in %s', i, query_vectors.shape[0], time.time() - t)
        del topk
        del d
        del idx
    logging.info('Done! %s', time.time() - t)
    return distances, indices

def knn_cdist(query_vectors, base_vectors, batch_size=10000, K=50):
    t = time.time()
    distances = np.zeros((query_vectors.shape[0], K), dtype=np.float32)
    indices = np.zeros((query_vectors.shape[0], K), dtype=np.int32)
    for i in range(0, query_vectors.shape[0], batch_size):
        res = np.matmul(query_vectors[i:(i+batch_size)],base_vectors.T)
        idx = np.argpartition(res, -K, axis=1)[:,-K:]
        d = res[np.arange(idx.shape[0])[:,None], idx]
        distances[np.arange(i, i + d.shape[0])] = d.astype(np.float32)
        indices[np.arange(i, i + d.shape[0])] = idx.astype(np.int32)
        logging.info('Finished %s out of %s in %s', i, query_vectors.shape[0], time.time() - t)
        del d
        del idx
    logging.info('Done! %s', time.time() - t)
    return distances, indices

def build_graph_from(flgs, seed):
    logging.info('Building input graph')
    gt = time.time
    np.random.seed(seed)

    # Load
    logging.info('Loading data from %s with format %s', flgs.graph_input, flgs.graph_input_format)
    t = gt()
    point_ids, point_labels, all_vectors = load(flgs.graph_input, flgs.graph_input_format, flgs.unit_norm)

    samples = []

    graph_num_neighbors = flgs.graph_num_neighbors if flgs.graph_num_neighbors > 0 else flgs.num_points

    # random classes
    unique_labels = np.unique(point_labels)
    if flgs.graph_num_classes != -1:
        C = flgs.graph_num_classes
    else:
        C = np.ceil(flgs.num_points * flgs.graph_percent_of_points_for_classes).astype(np.int32)

    C = np.minimum(C, len(unique_labels))

    logging.info('selected_labels = np.random.choice(%s, %s, replace=False)', len(unique_labels), C)

    selected_labels = np.random.choice(unique_labels, C, replace=False)

    vectors = all_vectors[np.isin(point_labels, selected_labels)].astype(np.float32)
    labels = point_labels[np.isin(point_labels, selected_labels)]
    pids = point_ids[np.isin(point_labels, selected_labels)]
    logging.info('np.random.choice(%s, %s, replace=False)', vectors.shape[0], flgs.num_points)
    random_points = np.random.choice(np.arange(vectors.shape[0]), flgs.num_points, replace=False)
    vectors = vectors[random_points]
    labels = labels[random_points]
    pids = pids[random_points]
    logging.info('Selected points: %s', str(pids))

    load_t = gt() - t
    logging.info('Finished loading data from %s with format %s in %s seconds', flgs.graph_input, flgs.graph_input_format, load_t)
    logging.info('Number of vectors (before unique) %s', vectors.shape[0])
    logging.info('Dim of vectors %s', vectors.shape[1])
    logging.info('Starting to build k=%s nn graph', graph_num_neighbors)
    t = gt()
    sims, neighbors = knn_cdist(vectors, vectors, K=graph_num_neighbors, batch_size=flgs.batch_size)
    query_t = gt() - t
    logging.info('NN time %s', query_t)

    dists = sims
    if flgs.build_cc_graph:
        dists -= np.mean(dists)
    logging.info('min dists %s', np.min(dists))
    logging.info('max dists %s', np.max(dists))
    i = np.repeat(np.arange(vectors.shape[0]), graph_num_neighbors, axis=0)
    j = np.squeeze(neighbors.reshape((-1, 1)))
    mat = coo_matrix((np.squeeze(dists.reshape(-1, 1))[i != j], (i[i != j], j[i != j])),
                     shape=(vectors.shape[0], vectors.shape[0]))
    mat = mat.tolil()
    ii, jj = mat.nonzero()
    mat[jj, ii] = mat[ii, jj]
    mat = mat.tocoo()
    mat.eliminate_zeros()
    samples.append((mat, (pids, labels, vectors)))
    logging.info('Finished computing graph.')
    return mat.tolil(), pids, labels, vectors
