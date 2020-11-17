.. _bede_tensorflow:

TensorFlow
==========

.. sidebar:: TensorFlow

   :URL: https://www.tensorflow.org/

TensorFlow is an open source software library for numerical computation using data flow graphs.
Nodes in the graph represent mathematical operations,
while the graph edges represent the multidimensional data arrays (tensors) communicated between them.
The flexible architecture allows you to deploy computation to
one or more CPUs or GPUs in a desktop, server, or mobile device
with a single API.
TensorFlow was originally developed by researchers and engineers working on the Google Brain Team
within Google's Machine Intelligence research organization
for the purposes of conducting machine learning and deep neural networks research,
but the system is general enough to be applicable in a wide variety of other domains as well.

About TensorFlow on Bede
------------------------

.. note::
   GPU must be requested in order to enable GPU acceleration by adding the flag e.g. ``--gpus=1`` to the scheduler command or job script.
   See :ref:`bede_scheduler` for more information.

Bede has a locally installed IBM `Watson Machine Learning Community Edition (WML CE) <https://developer.ibm.com/linuxonpower/deep-learning-powerai/releases/>`_ Anaconda channel that provides versions of Tensorflow, PyTorch and their dependencies especially built for the POWER architecture. 

Installation in Home Directory - GPU Version
--------------------------------------------

First request an interactive session, e.g. see :ref:`bede_scheduler`.

Then GPU version of TensorFlow can be installed by the following ::

    # Load the conda module
    module load Anaconda3/2020.02

    # Adds the local WML CE channel to the conda search path
    conda config --prepend channels file:///opt/software/apps/ibm_wmlce/wmlce-1.7.0-mirror/

    # Create an conda virtual environment e.g. named 'tensorflow'
    conda create -n tensorflow python=3.6

    # Activate the 'tensorflow' environment
    source activate tensorflow

    # Install GPU version of TensorFlow
    conda install tensorflow

To install a version of ``tensorflow`` other than the latest version
you should specify a version number when running ``conda install`` i.e. ::

   pip install tensorflow=<version_number>

To search for available versions use ``conda search`` i.e. ::

    conda search tensorflow

**Every Session Afterwards and in Your Job Scripts**

Every time you use a new session or within your job scripts, the modules must be loaded and Conda must be activated again.
Use the following command to activate the Conda environment with TensorFlow installed: ::

   module load Anaconda3/2020.02
   source activate tensorflow

Testing your TensorFlow installation
------------------------------------

You can test that TensorFlow is running on the GPU with the following python code ::

   import tensorflow as tf
   # Creates a graph
   #If using CPU, replace /device:GPU:0 with /cpu:0
   with tf.device('/device:GPU:0'):
      a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
      b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
      c = tf.matmul(a, b)
   # Creates a session with log_device_placement set to True.
   sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
   # Runs the op.
   print(sess.run(c))

Which should give the following results: ::

	[[ 22.  28.]
	 [ 49.  64.]]

