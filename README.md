# Learning-Based Model Predictive Control for Piecewise Affine Systems with Feasibility Guarantees

[![Source Code License](https://img.shields.io/badge/license-GPL-blueviolet)](https://github.com/SamuelMallick/mpcrl-vehicle-gears/blob/main/LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.13-green.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


This repository contains the source code used to produce the results obtained in [Learning-Based Model Predictive Control for Piecewise Affine Systems
with Feasibility Guarantees](https://arxiv.org/abs/2412.00490) published in [ECC 2025](https://ecc25.euca-ecc.org/).

In this work we propose a learning-based model predictive controller for piecewise affine systems.

If you find the paper or this repository helpful in your publications, please consider citing it.

```bibtex
@article{mallick2025learning,
  title={Learning-Based Model Predictive Control for Piecewise Affine Systems with Feasibility Guarantees},
  author={Mallick, Samuel and Dabiri, Azita and De Schutter, Bart},
  journal={arXiv preprint arXiv:2412.00490},
  year={2025}
}
```

---

## Installation

The code was created with `Python 3.12`. To access it, clone the repository

```bash
git clone https://github.com/SamuelMallick/supervised-learning-pwa-mpc
cd supervised-learning-pwa-mpc
```

and then install the required packages by, e.g., running

```bash
pip install -r requirements.txt
```

### Structure

- The scripts used to generate the results in the paper are found in **`examples/paper_2024`**.
- The core code for the approach is in the package source code **`src`**.

## License

The repository is provided under the GNU General Public License. See the [LICENSE](https://github.com/SamuelMallick/supervised-learning-pwa-mpc/blob/main/LICENSE) file included with this repository.

---

## Author

[Samuel Mallick](https://www.tudelft.nl/staff/s.h.mallick/), PhD Candidate [s.mallick@tudelft.nl | sam.mallick.97@gmail.com]

> [Delft Center for Systems and Control](https://www.tudelft.nl/en/3me/about/departments/delft-center-for-systems-and-control/) in [Delft University of Technology](https://www.tudelft.nl/en/)

> This research is part of a project that has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme ([Grant agreement No. 101018826 - CLariNet](https://cordis.europa.eu/project/id/101018826)).

Copyright (c) 2025 Samuel Mallick.

Copyright notice: Technische Universiteit Delft hereby disclaims all copyright interest in the program “mpcrl-vehicle-gearse” (Learning-Based Model Predictive Control for Piecewise Affine Systems with Feasibility Guarantees) written by the Author(s). Prof. Dr. Ir. Fred van Keulen, Dean of 3mE.