
import pickle
import os
import logging
import numpy as np
import time
import wandb
import sys


from absl import flags
from absl import logging
from absl import app

from AstarTrellis.iter_trellis_approx import IterCCTrellis, IterJetTrellis
# from .iter_trellis3 import all_two_partitions, k_random_2_cuts, top_k_of_n_2_cuts



NleavesMin=15
filename = "test_" + str(NleavesMin) + "_jets_gt_BS.pkl"
# filename = "test_" + str(NleavesMin) + "_jets.pkl"
powerset = 2**(NleavesMin)

# flags.DEFINE_string('trellis_class', 'IterJetTrellis', 'Type of Algorithm')
flags.DEFINE_string('trellis_class', 'Approx_IterJetTrellis', 'Type of Algorithm')

flags.DEFINE_string("wandb_dir", "/Users/sebastianmacaluso/Documents/A_star", "wandb directory - If running seewp process, run it from there")
flags.DEFINE_string('dataset_dir', "../../data/Ginkgo/input/", "dataset dir ")
flags.DEFINE_string('dataset', filename, 'dataset filename')
flags.DEFINE_integer('max_steps', 50000, 'Maximum number of steps')
flags.DEFINE_integer('all_pairs_max_size', 8, 'Maximum number of elements of a node to run the exact algorithm - switch to approx. algo for more elements')
flags.DEFINE_multi_integer('num_tries', [15,5], '')
flags.DEFINE_boolean("exact_heuristic_proof", False, "If true run algorithm with exact heuristic")
flags.DEFINE_boolean("approx_heuristic", True, "If true run algorithm with approximate heuristic, if False then run it with a supposedly exact heuristic check on plots with up to 9 elements but with no proof")


flags.DEFINE_integer('max_nodes', powerset + 10, 'nodes')
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
    wandb.log({'trellis_class': FLAGS.trellis_class,
               'child_func': FLAGS.child_func,
               'propagate_values_up': FLAGS.propagate_values_up})

    logging.info("Num tries = %s", FLAGS.num_tries)

    # logging.info('FLAGS.output = ',FLAGS.output)
    # logging.info("wandb.env.SWEEP_ID =", wandb.env.SWEEP_ID)
    # outprefix = os.path.join(FLAGS.output, wandb.env.get_project(), os.environ.get(wandb.env.SWEEP_ID, 'solo'),
    #                          wandb.env.get_run())
    # logging.info('outprefix is %s', outprefix)
    # os.makedirs(outprefix, exist_ok=True)
    np.random.seed(FLAGS.seed)

    # gt_jets = load_jets()
    gt_jets, BS_jets = load_jets()

    times=[]
    MAP = []
    steps =[]
    nodes_explored = []
    Ntrees = []
    for i in range(1,2):
        gt_jet = gt_jets[i]
        BS_jet = BS_jets[i]
        logging.info("Truth log LH = %s",sum(gt_jet["logLH"]))

        logging.info("test jet loaded ")
        logging.info('Running on a dataset (%s) with %s points ', FLAGS.dataset, gt_jet["leaves"].shape[0])



        startTime = time.time()

        # if FLAGS.trellis_class == 'IterJetTrellis':
        trellis = IterJetTrellis(exact_heuristic_proof = FLAGS.exact_heuristic_proof,
                                approx_heuristic = FLAGS.approx_heuristic,
                                 leaves=gt_jet['leaves'],
                                 propagate_values_up=FLAGS.propagate_values_up,
                                 max_nodes=FLAGS.max_nodes,
                                 min_invM=gt_jet['pt_cut'],
                                 Lambda=gt_jet['Lambda'],
                                 LambdaRoot=gt_jet['LambdaRoot'])

        trellis.Add_BeamSearchHC( BS_jet,
                         BS_jet['root_id'],
                         trellis.clusters[trellis.root])

            # #Define _get_children as all 2 partitions
            # if FLAGS.child_func == 'all_two_partitions':
            #     # _get_children = all_two_partitions
            #
            #     trellis._get_children = all_two_partitions
        logging.info("gt jet LH = %f | BS jet LH= %s ", sum(gt_jet["logLH"]), sum(BS_jet["logLH"]))
        # logging.info("gt jet LH = %f  ", sum(gt_jet["logLH"]))

        if FLAGS.trellis_class == 'Approx_IterJetTrellis':
            pass
            # BSO_jet = fill_BSList(gt_jet, Nbest=1)[0]
            # logging.info("gt jet = %f", sum(gt_jet["logLH"]))
            # logging.info("gt jet LH = %f | BS jet LH= %s | BSO jet LH= %s", sum(gt_jet["logLH"]), sum(BS_jet["logLH"]),sum(BSO_jet["logLH"]))
            # logging.info("gt jet LH = %f | BS jet LH= %s ", sum(gt_jet["logLH"]), sum(BS_jet["logLH"]))
            # logging.info("BS jet = %s", BS_jet)


        else:
            raise Exception("Unknown trellis %s" % FLAGS.trellis_class)




        en_t = time.time()
        # run search.
        # if FLAGS.trellis_class == 'IterJetTrellis':
        hc, f , step = trellis.execute_search(num_matches=FLAGS.num_repeated_map_values, max_steps =int(FLAGS.max_steps), all_pairs_max_size=int(FLAGS.all_pairs_max_size), num_tries= list(FLAGS.num_tries ))

        endTime = time.time() - startTime
        logging.info(f'init time = {en_t - startTime}')
        logging.info(f'search time = {time.time() - en_t}')
        logging.info(f'total time = {endTime}')

        # wandb.log({'a_star_map': f})
        logging.info("-------------------------------------------")
        logging.info(f'FINAL HC:{hc}')
        logging.info(f'FINAL f ={- f}')
        logging.info("Number of nodes explored =  %s", trellis.nodes_explored)
        logging.info("Number of trees explored =  %s", trellis.Ntrees)


        times.append(endTime)
        MAP.append( - f)
        steps.append(step)
        nodes_explored.append(trellis.nodes_explored)
        Ntrees.append(trellis.Ntrees)

    logging.info("==============================================")
    logging.info(f"Times = {times}")
    logging.info(f"MAP values ={MAP}")
    logging.info(f"Steps = {steps}")
    logging.info("Number of nodes explored =  %s", nodes_explored)
    logging.info("Number of trees explored =  %s", Ntrees)

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





def fill_BSList(input_jet, Nbest=1):
    """ Run the Beam search algorithm (algorithm where when the logLH of 2 or more trees is the same, we only keep one of them) over a list  of sets of input jets.
        Args: input jets
        returns: clustered jets
                     jets logLH
    """

    startTime = time.time()
    BSO_jetsList = []

    a = time.time()

    # for i, truth_jet in enumerate(input_jets):

        # if i % 50 == 0:
        #     print(" # of reclustered jets = ", i, "; Partial time = ", time.time() - a)
        #     #                 print("PARTIAL TIME = ",time.time() -a)
        #     a = time.time()
    logging.info("tree dict = %s ", input_jet)

    N = len(input_jet["leaves"])

    BSO_jetsList.append(BSO.recluster(
        input_jet,
        beamSize=min(3 * N, np.asarray(N * (N - 1) / 2).astype(int)),
        delta_min=input_jet["pt_cut"],
        lam=float(input_jet["Lambda"]),
        N_best=Nbest,
        visualize = False,
    )[0]
                            )

    # print("TOTAL TIME = ", time.time() - startTime)

    # BSO_jetsListLogLH = [sum(jet["logLH"]) for jet in BSO_jetsList]

    return BSO_jetsList



if __name__ == '__main__':
    app.run(main)
