import tensorflow as tf
import numpy as np
import itertools
import __main__ as main
from data_reader import Data
import argparse
import autoenc_shared as nn

main_dataset = Data(filenames=["../data/batched/batch{}.txt".format(i) for i in range(26)])
supporting_datasets = Data(["../data/80955_filtered.txt"],
	batch_size=50,
	padding_size=5)



def runner(_):
	model = [nn.Model(main_dataset)]

	model.append(nn.Model(supporting_datasets, 5, 1))

	cluster = tf.train.ClusterSpec({"ps": ["localhost:40000"], "worker": ["localhost:40001", "localhost:40002"]})

	if FLAGS.job_name == "ps":
		server = tf.train.Server(cluster, job_name="ps", task_index=0)
		server.join()
	elif FLAGS.job_name == "worker":
		nn.sess_runner(model, cluster)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.register("type", "bool", lambda v: v.lower() == "true")

	parser.add_argument(
	  "--job_name",
	  type=str,
	  default="",
	  help="One of 'ps', 'worker'"
	)

	parser.add_argument(
	  "--task_index",
	  type=int,
	  default=0,
	  help="Index of task within the job"
	)
	FLAGS, unparsed = parser.parse_known_args()


	tf.app.run(main=runner)




