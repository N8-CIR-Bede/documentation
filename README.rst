##################
Bede Documentation
##################

This is a collection of useful pieces of documentation for Bede. User contributions are encouraged.

.. image:: https://readthedocs.org/projects/bede-documentation/badge/?version=latest
  :target: https://bede-documentation.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status


*****************
How to Contribute
*****************

To contribute to this documentation, first create a fork of the repository on GitHub and clone it to your local machine, see `Fork a Repo <https://help.github.com/articles/fork-a-repo/>`_ for the GitHub documentation on this process.

Once you have cloned your fork, you will need to install the dependencies to be able to build the documentation. See the instructions below for how to achieve this.

Once you have made your changes, commited them and pushed them to your fork on GitHub you will need to `Open a Pull Request <https://help.github.com/articles/using-pull-requests/>`_. All changes to the repository should be made through Pull Requests, including those made by the people with direct push access.
Using feature branches is recommended.


***********************
Installing Dependencies
***********************

This documentation requires ``python`` and the python packages ``sphinx``, ``sphinx-autobuild`` and ``sphinx-bootstrap-theme``, as listed in ``requirements.txt``.
This would be typically done using a `Python Virtual Environment <https://docs.python.org/3/tutorial/venv.html>`_, or `conda <https://docs.conda.io/en/latest/>`_


Using a python ``venv``
=======================

.. code-block:: console

    mkdir -m 700 -p ~/.venvs
    python3 -m venv ~/.venvs/bede
    source ~/.venvs/bede/bin/activate
    pip install -r requirements.txt


Using conda (windows)
=====================

From a conda-enabled terminal:

.. code-block:: console

    conda create --name bede python=3
    conda activate bede
    pip install -r requirements.txt


**************************
Building the documentation
**************************

To build the HTML documentation run the following from a shell with the ``bede`` environment enabled:

.. code-block:: console

    make html

Or if you don't have the ``make`` utility installed on your machine then build with *sphinx* directly:

.. code-block:: console

    sphinx-build -W . ./html



Continuous build and serve
==========================

The package `sphinx-autobuild <https://github.com/GaretJax/sphinx-autobuild>`_ provides a watcher that automatically rebuilds the site as files are modified.

To start the autobuild process, from a shell with the ``bede`` environment enabled run: 

.. code-block:: console

    make livehtml

Or if you don't have the ``make`` utility installed on your machine then build with *sphinx-autobuild* directly:

.. code-block:: console

    sphinx-autobuild . ./html

The application also serves up the site at port ``8000`` by default at http://localhost:8000.


***********************************
Making Changes to the Documentation
***********************************

The documentation consists of a series of `reStructured Text <http://sphinx-doc.org/rest.html>`_ files which have the ``.rst`` extension. These files are then automatically converted to HTML and combined into the web version of the documentation by sphinx. It is important that when editing the files the syntax of the rst files is followed.


If there are any errors in your changes the build will fail and the documentation will not update, you can test your build locally by running ``make html``. The easiest way to learn what files should look like is to read the ``rst`` files already in the repository.


The docs use the `Sphinx Bootstrap Theme <https://github.com/ryan-roemer/sphinx-bootstrap-theme>`_ with customisations to match the N8 brand guidelines and to play nicely with ReadTheDocs. It should be possible to add new subpages to any of the main sections, but adding any further top-level headings would require changes to the custom CSS to ensure the menu does not break.
