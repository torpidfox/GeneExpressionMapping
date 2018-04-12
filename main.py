import tensorflow as tf
import numpy as np
import itertools
import __main__ as main
from logger import Logger
import autoenc_shared as nn
from multiprocessing import Process
from time import sleep
import argparse
import sys

gene_count = 2054
num_hidden_1 = gene_count // 10 # dunno yet
num_input =  gene_count


tasks = range(1)
bathes_size = [20]
FLAGS = None
init_param = [[0, ["../data/batched/batch1.txt"], 20, gene_count, 10], [1, ["../data/batched/batch2.txt"], 10, gene_count, 10]]


def runner(_):
	cluster = tf.train.ClusterSpec({"ps": ["localhost:40000"], "worker": ["localhost:40001", "localhost:40002"]})

	if FLAGS.job_name == "ps":
		server = tf.train.Server(cluster, job_name="ps", task_index=0)
		server.join()
	elif FLAGS.job_name == "worker":
		network = nn.Model(cluster, *init_param[FLAGS.task_index])
		network.sess_runner()
		# def ps_proc(cluster_spec):
# 	shared_params_holder = tf.train.Server(cluster, job_name="ps", task_index=0)
# 	shared_params_holder.join()


# ps_proc = Process(target=ps_proc, args=(cluster, ), daemon=True)

if __name__ == '__main__':
	#w1_proc = Process(target=models[0].sess_runner, args=(), daemon=True)
	parser = argparse.ArgumentParser()
	parser.register("type", "bool", lambda v: v.lower() == "true")
	# Flags for defining the tf.train.ClusterSpec
	# parser.add_argument(
	#   "--ps_hosts",
	#   type=str,
	#   default="",
	#   help="Comma-separated list of hostname:port pairs"
	# )
	# parser.add_argument(
	#   "--worker_hosts",
	#   type=str,
	#   default="",
	#   help="Comma-separated list of hostname:port pairs"
	# )
	parser.add_argument(
	  "--job_name",
	  type=str,
	  default="",
	  help="One of 'ps', 'worker'"
	)
	# Flags for defining the tf.train.Server
	parser.add_argument(
	  "--task_index",
	  type=int,
	  default=0,
	  help="Index of task within the job"
	)
	FLAGS, unparsed = parser.parse_known_args()


	tf.app.run(main=runner)

	#w2_proc = Process(target=models[1].sess_runner, args=(), daemon=True)
	#ps_proc.start()
	#w1_proc.start()
	#w2_proc.start()



