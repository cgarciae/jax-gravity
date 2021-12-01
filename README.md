# JAX Gravity
This repo contains a demo of how to use JAX to create a simple gravity simulation. It uses JAX's experimental `ode` package to solve the differential equation.

![gravity-animation](animation.gif)

One cool thing about this demo is that it creates a function called `gravity` that only calculates the force between two bodies, and then uses `jax.vmap` twice to transform it into a function that calculates the force between all pairs of bodies.

## Instalation

### pip
```bash
pip install -r requirements.txt
```

### poetry
```bash
poetry install
```

## Usage
Upon running running the following command, you should the animation of the simulation.

```bash
python main.py
```

Change the parameters for fun :)