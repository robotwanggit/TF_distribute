#https://raw.githubusercontent.com/robotwanggit/hello-world/master/ssgd.py
""" synchronous SGD"""
from __future__ import print_function
import tensorflow as tf
import argparse
import time
import os

flags = tf.app.flags
flags.DEFINE_string("ps_hosts", "localhost:2222",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None, "job name: worker or ps")
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
flags.DEFINE_integer("replicas_to_aggregate", 2,
                     "Number of replicas to aggregate before parameter update"
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")					 
FLAGS = flags.FLAGS

log_dir = '/logdir'


def main(unparsed):
	#config
	config = tf.ConfigProto(log_device_placement = False)
	
	#server setup
	ps_spec = FLAGS.ps_hosts.split(',')
	worker_spec = FLAGS.worker_hosts.split(',')
	cluster = tf.train.ClusterSpec({"ps":ps_spec, "worker":worker_spec})
	
	if FLAGS.job_name == 'ps':
		server = tf.train.Server(cluster, job_name = 'ps', task_index = \
		FLAGS.task_index, config = config)
		server.join()
		
	else:
		is_chief = (FLAGS.task_index ==0)
		server = tf.train.Server(cluster, 
			job_name = 'worker',
			task_index = FLAGS.task_index,
			config = config)
		
		#Graph
		worker_device = "/job:%s/task:%d/cpu:0"%(FLAGS.job_name, FLAGS.task_index)
		with tf.device(tf.train.replica_device_setter(ps_tasks = 1, worker_device = worker_device)):
			a = tf.Variable(tf.constant(0., shape = [2]), dtype = tf.float32)
			b = tf.Variable(tf.constant(0., shape = [2]), dtype = tf.float32)
			
			c = a + b
			global_step = tf.Variable(0, dtype = tf.int32, trainable = False, name = 'global_step')
			target = tf.constant(100., shape = [2], dtype = tf.float32)
			loss = tf.reduce_mean(tf.square(c - target))
			
			#create an optimizer then wrap it with SynchReplicasOptimizer
			optimizer = tf.train.GradientDescentOptimizer(.0001)
			optimizer1 = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate = FLAGS.replicas_to_aggregate, total_num_replicas = 2)
			opt = optimizer1.minimize(loss, global_step = global_step)
			
		#session
		sync_replicas_hook = optimizer1.make_session_run_hook(is_chief)
		stop_hook = tf.train.StopAtStepHook(last_step = 10)
		hooks = [sync_replicas_hook, stop_hook]
			
		#Monitored Training Session
		sess = tf.train.MonitoredTrainingSession(master = server.target,
				is_chief = is_chief,
				config = config,
				hooks = hooks,
				stop_grace_period_secs = 10)
				
		print("Starting training on worker %d "%FLAGS.task_index)
		while not sess.should_stop():
			_, r, gs = sess.run([opt, c, global_step])
			print(r, 'step:', gs, 'worker:', FLAGS.task_index)
			if is_chief: time.sleep(1)
			time.sleep(1)
		print('Done', FLAGS.task_index)
		
		time.sleep(10)
		sess.close()
		print("Session from worker %d closed cleanly"%FLAGS.task_index)
		
if __name__ == "__main__":
	print(FLAGS.task_index)
	tf.app.run()