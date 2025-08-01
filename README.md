# NeuralDEM

[[`Project Page`](https://emmi-ai.github.io/NeuralDEM)] 
[[`Paper`](https://arxiv.org/abs/2411.09678)] 
[[`BibTeX`](https://github.com/emmi-ai/NeuralDEM#citation)]
[[`Hopper Tutorial`](https://colab.research.google.com/github/Emmi-AI/NeuralDEM/blob/main/tutorial_hopper.ipynb)]
[[`Fluidized Bed Tutorial`](https://colab.research.google.com/github/Emmi-AI/NeuralDEM/blob/main/tutorial_fluidized_bed.ipynb)]

Run the tutorials direclty in google colab:
- [hopper](https://colab.research.google.com/github/Emmi-AI/NeuralDEM/blob/main/tutorial_hopper.ipynb)
- [fluidized bed](https://colab.research.google.com/github/Emmi-AI/NeuralDEM/blob/main/tutorial_fluidized_bed.ipynb)


Implementation of [NeuralDEM](https://arxiv.org/abs/2411.09678) containing:
- Hopper model implementation [src/neural_dem/hopper_model.py](https://github.com/Emmi-AI/NeuralDEM/blob/main/src/neural_dem/hopper_model.py)
- Fluidized bed reactor model implementation [src/neural_dem/fbed_model.py](https://github.com/Emmi-AI/NeuralDEM/blob/main/src/neural_dem/fbed_model.py) 

The implementation is accompanied with two tutorials notebook to showcase various aspects of our work:
- Data download and visualization
- Preprocessing data for an inference forward pass
- Instantiating models with pre-trained checkpoints from [huggingface](https://huggingface.co/EmmiAI/NeuralDEM)
- Visualize model predictions

We recommend to check it out yourself in an interactive google colab runtime:
- [hopper](https://colab.research.google.com/github/Emmi-AI/NeuralDEM/blob/main/tutorial_hopper.ipynb)
- [fluidized bed](https://colab.research.google.com/github/Emmi-AI/NeuralDEM/blob/main/tutorial_fluidized_bed.ipynb)

# Further resources

If our work sparked your interest, check out our other works and resources!

## Papers

- [UPT](https://arxiv.org/abs/2402.12365)
- [NeuralDEM](https://arxiv.org/abs/2411.09678)
- [AB-UPT](https://arxiv.org/abs/2502.09692)


## Videos/Visualizations

- [UPT overview](https://youtu.be/mfrmCPOn4bs)
- [NeuralDEM project page](https://emmi-ai.github.io/NeuralDEM)
- [Computer vision for large-scale neural physics simulations (covering UPT and NeuralDEM)](https://youtu.be/6lK2E8qn5bE)
- [AB-UPT demo](https://demo.emmi.ai/)


## Code

- [UPT code](https://github.com/ml-jku/UPT/)
- [UPT tutorial](https://github.com/BenediktAlkin/upt-tutorial)
- [AB-UPT code](https://github.com/Emmi-AI/AB-UPT)


# Citation

If you like our work, please consider giving it a star :star: and cite us

```
@article{alkin2024neuraldem,
    title={{NeuralDEM} - Real-time Simulation of Industrial Particulate Flows},
    author={Benedikt Alkin and Tobias Kronlachner and Samuele Papa and Stefan Pirker and Thomas Lichtenegger and Johannes Brandstetter},
    journal={arXiv preprint arXiv:2411.09678},
    year={2024}
}
```