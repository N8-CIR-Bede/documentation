##################
Bede Documentation
##################

This is a collection of useful pieces of documentation for Bede. User contributions are encouraged.

.. image:: https://readthedocs.org/projects/bede-documentation/badge/?version=latest
  :target: https://bede-documentation.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status
.. image:: https://github.com/N8-CIR-Bede/documentation/actions/workflows/ci.yml/badge.svg
  :target: https://github.com/N8-CIR-Bede/documentation/actions/workflows/ci.yml
  :alt: CI
.. image:: https://img.shields.io/badge/docs-bede--documentation.readthedocs.io-054C91
  :target: https://bede-documentation.readthedocs.io
  :alt: CI

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

This documentation requires ``python >= 3.9`` and a number of python packages as listed in ``requirements.txt``.
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

    conda create --name bede python=3.9
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


Testing Read the Docs extensions 
================================

When hosted on Read the Docs, additional Javascript is injected into the page(s) to add the version selector and ethical adverts.

To test this locally, define the environment variable ``MOCK_RTD`` locally and build the documentation. You may need to serve the content over a webserver to avoid CORS errors (i.e. use ``livehtml``).

.. code-block:: bash

   MOCK_RTD="True" make clean livehtml


***********************************
Making Changes to the Documentation
***********************************

The documentation consists of a series of `reStructured Text <http://sphinx-doc.org/rest.html>`_ files which have the ``.rst`` extension. These files are then automatically converted to HTML and combined into the web version of the documentation by sphinx. It is important that when editing the files the syntax of the rst files is followed.


If there are any errors in your changes the build will fail and the documentation will not update, you can test your build locally by running ``make html``. The easiest way to learn what files should look like is to read the ``rst`` files already in the repository.


The docs use the `Sphinx Book Theme <https://github.com/executablebooks/sphinx-book-theme>`_ with customisations to match the N8 brand guidelines.

***********************************
Accessibility Testing via ``pa11y``  
***********************************

To evaluate webpage accessibility, tools such as `pa11y <https://github.com/pa11y>`_ can be used to evaluate if accessibility guidelines are being met. 

After `installing pa11y-ci <https://github.com/pa11y/pa11y-ci#requirements>`__, and building the documentation locally it can be used to parse individual html files, or lists of html files.
Sphinx generates some html files which will fail accessibility tests, which are non-trivial to fix manually, so ignoring certain files is worthwhile.

Checking all generted html files can take a number of minutes.

.. code-block:: bash

   # Check the index page for accessibility issues
   pa11y-ci ./_build/html/index.html

   # Find and parse html files in _build/html excluding certain files which we cannot correct.
   pa11y-ci $(find _build/html -name "*.html" -and -not -path "*webpack*" -and -not -name "genindex.html" -and -not -name "search.html")

   # Produce Json output for subsequent parsing, i.e. to integrate into CI if desired.
   pa11y-ci --json $(find _build/html -name "*.html" -and -not -path "*webpack*" -and -not -name "genindex.html" -and -not -name "search.html") > pa11y-ci-report.json
