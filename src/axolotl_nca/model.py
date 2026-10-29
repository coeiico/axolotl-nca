"""
The model for axolotl-nca.

each cell holds 16 channels (RGBA plus 12 hidden). Fixed 3×3 Sobel and identity filters
let it sense its neighbours. Two 1×1 convolutions (128 hidden, ReLU) compute an update.
Cells update at random, and cells with low alpha count as dead.

forward() accepts return_cache=True and returns a dict of internal activations, and a
patch argument (or forward hooks) overwrites a named activation for interventions.
"""

# TODO: implement. See README.md for the plan.
