# Code for the project  "Causal Strategic Learning with Competitive Selection"

<p align="center">
<image src="https://github.com/muandet-lab/causal-strategic-learning-with-selection/blob/master/csl-diagram-1.png"/>
</p>



To instantiate the project environment, with conda:
```
conda create -n csl python=3.7
conda activate csl
python -m pip install -U -r requirements.txt
```

Or with Docker:
```
bash docker-env/launch.sh nb
```

To reproduce the experiments, please run `experiments.ipynb`
