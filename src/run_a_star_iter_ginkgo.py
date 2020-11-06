
import pickle
import os
import logging
import numpy as np
import time

import sys
# sys.path.append('../')

import wandb


from absl import flags
from absl import logging
from absl import app

from src.iter_trellis import IterCCTrellis, IterJetTrellis
from src.iter_trellis import all_two_partitions, k_random_2_cuts, top_k_of_n_2_cuts



NleavesMin=9

flags.DEFINE_string("wandb_dir", "/Users/sebastianmacaluso/Documents/A_star", "wandb directory - If running seewp process, run it from there")
flags.DEFINE_string('dataset_dir', "../data/Ginkgo/", "dataset dir ")
flags.DEFINE_string('dataset', "test_" + str(NleavesMin) + "_jets.pkl", 'dataset filename')
flags.DEFINE_integer('max_iter', 1000000000000, 'iterations')
flags.DEFINE_integer('max_nodes', 10000000, 'nodes')
flags.DEFINE_string('trellis_class', 'IterJetTrellis', 'dataset file')
flags.DEFINE_string('exp_name', 'AStar', 'name')
flags.DEFINE_string('output', 'exp_out', 'output directory')
flags.DEFINE_integer('num_points', 12, '')
flags.DEFINE_integer('seed', 42, '')
flags.DEFINE_string('child_func', 'all_two_partitions', 'function used to get children when initializing nodes')
flags.DEFINE_integer('num_repeated_map_values', 0, 'number of times the same MAP value is returned before halting') #This was for the approx. A*. Not implemented for Ginkgo. Set to 0.
flags.DEFINE_integer('propagate_values_up', 0, 'whether to propagate f,g,h values during trellis extension.')



FLAGS = flags.FLAGS

logging.set_verbosity(logging.INFO)

def main(argv):
    logging.info('Running with args %s', str(argv))

    wandb.init(project="%s" % (FLAGS.exp_name), dir=FLAGS.wandb_dir)
    wandb.config.update(flags.FLAGS)

    # logging.info('FLAGS.output = ',FLAGS.output)
    # logging.info("wandb.env.SWEEP_ID =", wandb.env.SWEEP_ID)
    # outprefix = os.path.join(FLAGS.output, wandb.env.get_project(), os.environ.get(wandb.env.SWEEP_ID, 'solo'),
    #                          wandb.env.get_run())
    # logging.info('outprefix is %s', outprefix)
    # os.makedirs(outprefix, exist_ok=True)
    np.random.seed(FLAGS.seed)

    gt_jets = load_jets()

    times=[]
    MAP = []
    steps =[]
    for i in range(1):
        gt_jet = gt_jets[i]

        logging.info("test jet loaded ")
        logging.info('Running on a dataset (%s) with %s points ', FLAGS.dataset, gt_jet["leaves"].shape[0])

        startTime = time.time()

        if FLAGS.trellis_class == 'IterJetTrellis':
            trellis = IterJetTrellis(leaves = gt_jet['leaves'],
                                     propagate_values_up=FLAGS.propagate_values_up,
                                     max_nodes=FLAGS.max_nodes,
                                     min_invM=gt_jet['pt_cut'],
                                     Lambda= gt_jet['Lambda'],
                                     LambdaRoot=gt_jet['LambdaRoot'])

            #Define _get_children as all 2 partitions
            if FLAGS.child_func == 'all_two_partitions':
                _get_children = all_two_partitions

            trellis._get_children = _get_children

        else:
            raise Exception("Unknown trellis %s" % FLAGS.trellis_class)


        wandb.log({'trellis_class': FLAGS.trellis_class,
                   'child_func': FLAGS.child_func,
                   'propagate_values_up': FLAGS.propagate_values_up})

        # run search.
        if FLAGS.trellis_class == 'IterJetTrellis':
            hc, f , step = trellis.execute_search(num_matches=FLAGS.num_repeated_map_values)

        endTime = time.time() - startTime
        wandb.log({'a_star_map': f})
        logging.info("-------------------------------------------")
        logging.info(f'FINAL HC:{hc}')
        logging.info(f'FINAL f ={f}')
        logging.info(f'total time = {endTime}')

        times.append(endTime)
        MAP.append(f)
        steps.append(step)

    logging.info("==============================================")
    logging.info(f"Times = {times}")
    logging.info(f"MAP values ={MAP}")
    logging.info(f"Steps = {steps}")

def load_jets():
    #
    # indir = "trellis/data/"
    indir=FLAGS.dataset_dir
    # indir="/Users/sebastianmacaluso/Dropbox/Documents/Physics_projects/simulator/A_starTrellis/hierarchical-trellis/data/"
    # in_filename = os.path.join(indir, "test_" + str(NleavesMin) + "_jets.pkl")
    in_filename = os.path.join(indir, FLAGS.dataset)
    with open(in_filename, "rb") as fd:
        gt_jets = pickle.load(fd, encoding='latin-1')
    return gt_jets


if __name__ == '__main__':
    app.run(main)
