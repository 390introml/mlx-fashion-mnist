# Demo for training Fashion MNIST autoencoder locally

Instructions

0. Install [uv](https://docs.astral.sh/uv/)
1. `uv venv`
2. `source .venv/bin/activate` 
3. `time uv run train.py`

Training should just take about a minute.

Questions:
1. How does batch size affect runtime?
2. How does batch size training stability?
3. Can you modify the code to plot the loss after each gradient descent step?
4. Can you have the code run using a different [optimizer](https://ml-explore.github.io/mlx/build/html/python/optimizers.html#)? What is an optimizer? How does your choice of optimizer affect training performance and training time?

