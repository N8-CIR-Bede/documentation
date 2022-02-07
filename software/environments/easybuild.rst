Easybuild
=========

.. note::

    Not currently recommended.

The central Easybuild modules are available when a user executes the
following command and then logs in again:

.. code-block:: bash

   echo easybuild > ~/.application_environment

A user can create their own Easybuild installation to supplement (or
override) the packages provided by the central install by:

.. code-block:: bash

   echo 'export EASYBUILD_INSTALLPATH=$HOME/eb' >> ~/.bash_profile
   echo 'export EASYBUILD_BUILDPATH=/tmp' >> ~/.bash_profile
   echo 'export EASYBUILD_MODULES_TOOL=Lmod' >> ~/.bash_profile
   echo 'export EASYBUILD_PARALLEL=8' >> ~/.bash_profile
   echo 'export MODULEPATH=$HOME/eb/modules/all:$MODULEPATH' >> ~/.bash_profile

Login again, and then:

.. code-block:: bash

   wget https://raw.githubusercontent.com/easybuilders/easybuild-framework/develop/easybuild/scripts/bootstrap_eb.py
   python bootstrap_eb.py $EASYBUILD_INSTALLPATH

Verify install by checking sensible output from:

.. code-block:: bash

   module avail   # should show an EasyBuild module under user's home directory
   module load EasyBuild
   which eb       # should show a path under the user's home directory

Software can now be installed into the new Easybuild area using
``eb <package>``

Project Easybuild installations can be created using a similar method.
In this case, a central module to add the project’s modules to a user’s
environment is helpful, and can be done on request.