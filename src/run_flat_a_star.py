
import pickle
import os
import logging
import numpy as np

from trellis.iter_trellis import IterCCTrellis
from trellis.iter_trellis import all_two_partitions, k_random_2_cuts, top_k_of_n_2_cuts
from trellis.tree import Tree
from trellis.build_graphs import build_graph_from
from trellis.flat import all_two_partitions_flat

import wandb
from sklearn.datasets import make_blobs

from absl import flags
from absl import logging
from absl import app

flags.DEFINE_integer('max_iter', 1000000000000, 'iterations')
flags.DEFINE_string('dataset', 'data/a_star/cc_aloi_samples', 'dataset file')
flags.DEFINE_string('trellis_class', 'DPMeansTrellis', 'dataset file')
flags.DEFINE_string('exp_name', 'AStar', 'name')
flags.DEFINE_string('output', 'exp_out', 'output directory')
flags.DEFINE_integer('tree_num', 0, '')
flags.DEFINE_integer('num_points', 12, '')
flags.DEFINE_integer('seed', 42, '')
flags.DEFINE_string('child_func', 'all_two_partitions', 'function used to get children when initializing nodes')
flags.DEFINE_integer('num_repeated_map_values', 0, 'number of times the same MAP value is returned before halting')
flags.DEFINE_integer('propagate_values_up', 0, 'whether to propagate f,g,h values during trellis extension.')

flags.DEFINE_float('opening_cost', 0.5, 'cost of opening a center, dpmeans')


flags.DEFINE_string('graph_input', 'blobs', 'The data file to build the graph from in either xcluster or npy format.')
flags.DEFINE_string('graph_input_format', 'xcluster', 'Input data format = {xcluster|npy}')
flags.DEFINE_integer('graph_num_neighbors', -1, 'The number of neighbors to give each point in the graph. -1 to do fully connected.')
flags.DEFINE_integer('unit_norm', 1, 'Whether or not to unit norm the vectors (should always be true)')
flags.DEFINE_integer('batch_size', 100, 'Batchsize when computing nearest neighbors.')
flags.DEFINE_integer('graph_num_classes', 3, 'Number of classes of points to use.')
flags.DEFINE_boolean('build_cc_graph', True, '')
flags.DEFINE_float('graph_percent_of_points_for_classes', 0.20, 'Number of classes of points to use.')

flags.DEFINE_integer('blobs_samples', 10, 'number of neighbors')
flags.DEFINE_integer('blobs_dim', 2, 'number of neighbors')
flags.DEFINE_integer('blobs_num_clusters', 3, 'number of neighbors')
flags.DEFINE_float('blobs_std', 0.05, 'number of neighbors')
flags.DEFINE_float('blobs_center_box_min', -1.0, 'number of neighbors')
flags.DEFINE_float('blobs_center_box_max', 1.0, 'number of neighbors')
flags.DEFINE_integer('blobs_center_seed', 5446, 'number of neighbors')

def create_blob_dataset(n_samples=100, n_features=2, centers=10, cluster_std=1.0,
                   center_box=(-10.0, 10.0), shuffle=True, seed=5446):
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std,
                   center_box=center_box, shuffle=shuffle, random_state=seed)
    return np.arange(X.shape[0]), y, X.astype(np.float32)


FLAGS = flags.FLAGS

logging.set_verbosity(logging.INFO)

def main(argv):
    logging.info('Running with args %s', str(argv))

    wandb.init(project="%s" % (FLAGS.exp_name))
    wandb.config.update(flags.FLAGS)
    outprefix = os.path.join(FLAGS.output, wandb.env.get_project(), os.environ.get(wandb.env.SWEEP_ID, 'solo'),
                             wandb.env.get_run())
    logging.info('outprefix is %s', outprefix)
    os.makedirs(outprefix, exist_ok=True)
    np.random.seed(FLAGS.seed)

    # for now just make random data
    if FLAGS.graph_input != 'blobs':
        _, point_ids, point_labels, points = build_graph_from(FLAGS, FLAGS.tree_num)
        dataset_name = os.path.basename(FLAGS.graph_input)
    else:
        point_ids, point_labels, points = create_blob_dataset(n_samples=FLAGS.blobs_samples, n_features=FLAGS.blobs_dim,
                        centers=FLAGS.blobs_num_clusters, cluster_std=FLAGS.blobs_std,
                        center_box=(FLAGS.blobs_center_box_min, FLAGS.blobs_center_box_max),
                        shuffle=True, seed=FLAGS.blobs_center_seed)
        dataset_name = 'blobs'

    from trellis.flat import DPMeansTrellis
    wandb.log({'trellis_class': FLAGS.trellis_class, 'tree_num': FLAGS.tree_num, 'dataset_name': dataset_name, 'num_points': FLAGS.num_points})

    if FLAGS.trellis_class == 'DPMeansTrellis':
        trellis = DPMeansTrellis(points, opening_cost=FLAGS.opening_cost, max_nodes=10000000)
        trellis._get_children = all_two_partitions_flat
        fc, f = trellis.execute_search()
    elif FLAGS.trellis_class == 'HACDPMeans':
        from scipy.spatial.distance import cdist
        from scipy.sparse import coo_matrix
        graph = cdist(points, points, metric='sqeuclidean')
        graph = graph.max() - graph
        tree = Tree(graph=coo_matrix(graph))
        hc, f = tree.execute_search()
        trellis = DPMeansTrellis(points, opening_cost=FLAGS.opening_cost, max_nodes=1000000)
        trellis.load_from_tree(tree, hc)
        trellis._get_children = lambda x: []
        fc, f = trellis.execute_search()
    elif FLAGS.trellis_class == 'HACDPMeans++':
        from scipy.spatial.distance import cdist
        from scipy.sparse import coo_matrix
        graph = cdist(points, points, metric='sqeuclidean')
        graph = graph.max() - graph
        tree = Tree(graph=coo_matrix(graph))
        hc, f = tree.execute_search()
        trellis = DPMeansTrellis(points, opening_cost=FLAGS.opening_cost, max_nodes=1000000)
        trellis.load_from_tree(tree, hc)
        trellis._get_children = all_two_partitions_flat
        fc, f = trellis.execute_search()

    wandb.log({'a_star_map': f})


if __name__ == '__main__':
    app.run(main)
