import tensorflow as tf
import numpy as np
import itertools
import __main__ as main
from data_reader_r import Data
from subset import PrivateDomain
import argparse
from model import sess_runner

main_dataset = Data(filenames=['../data/batch{}.txt'.format(i) for i in range(10)],
	batch_size=50,
	log=False,
	norm=False,
	exp=True,
	reuse=False)

supporting_dataset = Data(filenames=['../data/75140_m.txt'],
	batch_size=50,
	log=True,
	norm=False,
	exp=False,
	reuse=True)

# main_dataset = Data(["../data/schiza_formatted.txt"],
# 	batch_size=50,
# 	# additional_info="../data/schiza_ann.txt",
# 	# tags=[0, 1],
# 	reuse=True,
# 	log=True
# )

# supporting_dataset = Data(["../data/80655_formatted.txt"],
# 	batch_size=50,
# 	norm=False,
# 	log=True,
# 	reuse=True)

# check_dataset = Data(["../data/42546_tagged.txt"],
# 	batch_size=25,
# 	padding_size=2,
# 	additional_info="../data/42546_d.txt",
# 	valid=False,
# 	tags=[0, 1],
# 	log=True,
# 	reuse=True)



def runner(_):
	model = [PrivateDomain(main_dataset, delay=1, tagged=False)]

	model.append(PrivateDomain(supporting_dataset, delay=1, ind=1, tagged=False))

	#model.append(nn.Model(check_dataset, column = 'Disease', ind=2, classes=[0, 1]))

	cluster = tf.train.ClusterSpec({"ps": ["localhost:40000"], "worker": ["localhost:40001", "localhost:40002"]})

	if FLAGS.job_name == "ps":
		server = tf.train.Server(cluster, job_name="ps", task_index=0)
		server.join()
	elif FLAGS.job_name == "worker":
		sess_runner(model)
		#nn.sess_runner(model, ["red/model0.ckpt", "red/model1.ckpt", "red/model2.ckpt"])


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




