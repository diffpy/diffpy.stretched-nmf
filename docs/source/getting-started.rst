.. _getting-started:

Getting started
===============

``diffpy.stretched-nmf`` implements the stretched NMF algorithm for factorizing signal
sets while accounting for uniform stretching along the independent axis.

Installation
------------

The preferred method is conda:

.. code-block:: bash

   conda config --add channels conda-forge
   conda create -n diffpy.stretched-nmf_env diffpy.stretched-nmf
   conda activate diffpy.stretched-nmf_env

Alternatively, install from PyPI with pip:

.. code-block:: bash

   pip install diffpy.stretched-nmf

For source installs (after cloning the repo):

.. code-block:: bash

   pip install .

Quick check
-----------

Verify the CLI and Python import:

.. code-block:: bash

   diffpy.stretched-nmf --version
   python -c "import diffpy.stretched_nmf; print(diffpy.stretched_nmf.__version__)"

Basic usage
-----------

The main entry point is the ``SNMFOptimizer`` class. Provide a source matrix
with shape ``(length_of_signal, number_of_signals)`` and either ``n_components``
or ``init_weights``.

.. code-block:: python

   import numpy as np
   from diffpy.stretched_nmf.snmf_class import SNMFOptimizer

   rng = np.random.default_rng(0)
   # Example data: 200 points across 8 signals
   source_matrix = rng.random((200, 8))

   model = SNMFOptimizer(
       source_matrix,
       n_components=3,
       random_state=0,
   )

   model.fit(rho=0, eta=0)

   components = model.components_
   weights = model.weights_
   stretch = model.stretch_

Notes
-----

- ``rho`` controls the stretching penalty (set to ``0`` for no stretching).
- ``eta`` controls sparsity (start at ``0`` and tune after selecting ``rho``).

Next steps
----------

Browse the rest of the docs for release notes and license information.
