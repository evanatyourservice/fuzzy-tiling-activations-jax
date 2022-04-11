# Fuzzy Tiling Activations

A Simple Approach to Learning Sparse Representations Online

### Notes:

If eta is set to None, it defaults to (high - low) / n_tiles, which is 
fuzziness the same width as the tile.

If layer_norm_input is True, input is normalized before tiling and low 
and high are set to -3 and 3 respectively. If sigmoid_input is True, 
input is put through sigmoid first and low and high are set to 0 and 1 
respectively. Both can be True in which case sigmoid is after layer norm 
and low, high = 0, 1.

### Relevant papers:

**2019** [**Fuzzy Tiling Activations: A Simple Approach to Learning Sparse Representations Online**](https://arxiv.org/abs/1911.08068)
Yangchen Pan, Kirby Banman, Martha White

**2022** [**Investigating the Properties of Neural Network Representations in Reinforcement Learning**](https://arxiv.org/abs/2203.15955)
Han Wang, Erfan Miahi, Martha White, Marlos C. Machado, Zaheer Abbas, Raksha Kumaraswamy, Vincent Liu, Adam White
