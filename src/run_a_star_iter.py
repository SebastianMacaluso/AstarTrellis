
import pickle
import os
import logging
import numpy as np
import sys

# sys.path.append('../')

from trellis.iter_trellis import IterCCTrellis
from trellis.iter_trellis import all_two_partitions, k_random_2_cuts, top_k_of_n_2_cuts
from trellis.tree import Tree
from trellis.build_graphs import build_graph_from

import wandb


from absl import flags
from absl import logging
from absl import app

flags.DEFINE_integer('max_iter', 1000000000000, 'iterations')
flags.DEFINE_integer('max_nodes', 10000000, 'nodes')
flags.DEFINE_string('dataset', None, 'dataset file')
flags.DEFINE_string('trellis_class', 'IterCCTrellis', 'dataset file')
flags.DEFINE_string('exp_name', 'AStar', 'name')
flags.DEFINE_string('output', 'exp_out', 'output directory')
flags.DEFINE_integer('tree_num', 0, '')
flags.DEFINE_integer('num_points', 12, '')
flags.DEFINE_integer('seed', 42, '')
flags.DEFINE_string('child_func', 'all_two_partitions', 'function used to get children when initializing nodes')
flags.DEFINE_integer('num_repeated_map_values', 1, 'number of times the same MAP value is returned before halting')
flags.DEFINE_integer('propagate_values_up', 0, 'whether to propagate f,g,h values during trellis extension.')


flags.DEFINE_string('graph_input', None, 'The data file to build the graph from in either xcluster or npy format.')
flags.DEFINE_string('graph_input_format', 'xcluster', 'Input data format = {xcluster|npy}')
flags.DEFINE_integer('graph_num_neighbors', -1, 'The number of neighbors to give each point in the graph. -1 to do fully connected.')
flags.DEFINE_integer('unit_norm', 1, 'Whether or not to unit norm the vectors (should always be true)')
flags.DEFINE_integer('batch_size', 100, 'Batchsize when computing nearest neighbors.')
flags.DEFINE_integer('graph_num_classes', 3, 'Number of classes of points to use.')
flags.DEFINE_float('graph_percent_of_points_for_classes', 0.20, 'Number of classes of points to use.')
flags.DEFINE_boolean('build_cc_graph', True, '')


flags.DEFINE_integer('get_children_k', 5, '')
flags.DEFINE_integer('get_children_n', 3, '')
flags.DEFINE_integer('get_children_exact_size', 10, '')
flags.DEFINE_integer('get_children_num_tries', 2, '')
flags.DEFINE_integer('iter_approx_max_iter', 3, '')
flags.DEFINE_integer('search_cap_size', 40, '')


FLAGS = flags.FLAGS

logging.set_verbosity(logging.WARNING)

def main(argv):
    logging.info('Running with args %s', str(argv))


    wandb.init(project="%s" % (FLAGS.exp_name))
    wandb.config.update(flags.FLAGS)
    outprefix = os.path.join(FLAGS.output, wandb.env.get_project(), os.environ.get(wandb.env.SWEEP_ID, 'solo'),
                             wandb.env.get_run())
    logging.info('outprefix is %s', outprefix)
    os.makedirs(outprefix, exist_ok=True)
    np.random.seed(FLAGS.seed)

    # assert FLAGS.dataset is None or FLAGS.graph_input is None
    # if FLAGS.dataset:
    #     # load data
    #     with open('%s.pt%s' % (FLAGS.dataset, FLAGS.num_points) + ".pkl", "rb") as fd:
    #         graphs = pickle.load(fd, encoding='latin-1')
    #     # graphs = [x[0] for x in data]
    #     graph = graphs[FLAGS.tree_num][0].tolil()
    #     labels = graphs[FLAGS.tree_num][1][1]
    #     dataset_name = os.path.basename(FLAGS.dataset)
    # else:
    graph, _, labels, _ = build_graph_from(FLAGS, FLAGS.tree_num)
    dataset_name = os.path.basename(FLAGS.graph_input)

    logging.info('Running on a dataset (%s) with %s points from %s gt clusters', dataset_name, labels.shape[0],
                 np.unique(labels).shape[0])

    assert labels.shape[0] == FLAGS.num_points

    wandb.log({
        'tree_num': FLAGS.tree_num,
        'dataset_name': dataset_name,
        'num_points': FLAGS.num_points,
        'num_gt_clusters': labels.shape[0]
    })

    if FLAGS.trellis_class == 'IterCCTrellis' or FLAGS.trellis_class == 'GreedyIterCCTrellis':
        trellis = IterCCTrellis(graph, propagate_values_up=FLAGS.propagate_values_up, max_nodes=FLAGS.max_nodes)

        #Define _get_children as all 2 partitions or top k of n 2 cuts
        if FLAGS.child_func == 'all_two_partitions':
            _get_children = all_two_partitions
        else:
            def _get_children(elem):
                return top_k_of_n_2_cuts(elem, graph=trellis.graph, all_pairs_max_size=15)
        trellis._get_children = _get_children
        #
        if FLAGS.trellis_class == 'GreedyIterCCTrellis':
            tree = Tree(graph)
            hc, f = tree.execute_search()
            trellis.load_from_tree(tree, hc)

    elif FLAGS.trellis_class == 'Greedy':
        trellis = Tree(graph)
    elif FLAGS.trellis_class == 'Greedy++':
        tree = Tree(graph=graph)
        hc, f = tree.execute_search()
        trellis = IterCCTrellis(graph, propagate_values_up=FLAGS.propagate_values_up, max_nodes=FLAGS.max_nodes)
        trellis.scaffold_from_tree(tree, hc, FLAGS.propagate_values_up)
        trellis._get_children = all_two_partitions
        print('greedy tree', [i for i in trellis.clusters if i])
        hc, f = trellis.execute_search(num_matches=FLAGS.num_repeated_map_values)
    elif FLAGS.trellis_class == 'IterApprox':
        tree = Tree(graph=graph)
        hc, f = tree.execute_search()
        wandb.log({"search_f": f, "search_i": 0})
        trellis = IterCCTrellis(graph, propagate_values_up=FLAGS.propagate_values_up, max_nodes=FLAGS.max_nodes)
        trellis.scaffold_from_tree(tree, hc, FLAGS.propagate_values_up)
        def _get_children(elem):
            return top_k_of_n_2_cuts(elem, graph=trellis.graph,
                                     all_pairs_max_size=FLAGS.get_children_exact_size,
                                     k= FLAGS.get_children_k if FLAGS.get_children_k != -1 else None,
                                     n = FLAGS.get_children_n if FLAGS.get_children_n != -1 else None,
                                     num_tries=lambda x: x*FLAGS.get_children_num_tries,
                                     search_cap_size=FLAGS.search_cap_size)
        trellis._get_children = _get_children
        print('greedy tree', [i for i in trellis.clusters if i])
        hc, f = trellis.execute_search(num_matches=FLAGS.num_repeated_map_values, max_iter=FLAGS.iter_approx_max_iter)
    else:
        raise Exception("Unknown trellis %s" % FLAGS.trellis_class)

    wandb.log({'trellis_class': FLAGS.trellis_class,
               'child_func': FLAGS.child_func,
               'propagate_values_up': FLAGS.propagate_values_up})

    # run search.
    if FLAGS.trellis_class == 'IterCCTrellis' or FLAGS.trellis_class == 'GreedyIterCCTrellis' :
        hc, f = trellis.execute_search(num_matches=FLAGS.num_repeated_map_values)
    elif FLAGS.trellis_class == 'Greedy':
        hc, f = trellis.execute_search()

    wandb.log({'a_star_map': f})
    print('FINAL HC:', hc)
    print('FINAL f:', f)
    # magic_num = 19
    # print('looking at', trellis.clusters[magic_num])
    # for f,g,h,l,r in trellis.pq[magic_num][:3]:
    #     print('tttt', f, g, h, trellis.clusters[l], trellis.clusters[r])
    #     print('uu', trellis.up_to_date_fgh[magic_num][frozenset((l, r))])
    # for f, g, h, ch_l, ch_r in trellis.pq[trellis.root][:10]:
    #     print('f, g, h, ch_l, ch_r', f, g, h, trellis.clusters[ch_l], trellis.clusters[ch_r])
    # print('XXX')
    # print()
    # f, g, h, ch_l, ch_r in trellis.pq[trellis.root][1]
    # print('X: f, g, h, ch_l, ch_r', f, g, h, trellis.clusters[ch_l], trellis.clusters[ch_r])
if __name__ == '__main__':
    app.run(main)
