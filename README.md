# Python codes for testing solutions to aggregation-type equations.

## Introduction

This repo contains a Python module for plotting the solution to the aggregation-diffusion equation and computing the convergence errors either when the viscosity tends to 0 or when the discretization tends to 0.

To refactor:
- A script which does the same thing for stationary solutions (requires the MpMath package)
- A illustration script to display the computed convergence order.

Much of the code has been written with the precious help of Benoit Fabrèges.

## Installation

### Prerequisites

- python >= 3.10
- pip
- (preferably) an installed virtual environment

### Install requirements

```bash
pip install -r requirements.txt
```

### Clone the repo

```bash
git clone https://github.com/strantien/aggregation
```

## Usage

Prompting all the possible parameters:

```bash
python -m aggdiff --help
```

Running a simple simulation of a solution to an aggregation-diffusion equation

```bash
python -m aggdiff -simu
```

Computing convergence errors w.r.t dx and saving both the errors and the values of rho as text files in a (new) directory:

```bash
python -m aggdiff -cvg --convergence-parameter dx
```

Same w.r.t epsilon:

```bash
python -m aggdiff -cvg --convergence-parameter eps
```

