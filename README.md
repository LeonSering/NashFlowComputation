# Computing Nash Flows Over Time and Thin Flows
The *Nash Flow Computation Tool* (NFC) allows the computation of normalized thin flows with resetting and consequently Nash
flows over time with or without spillback. 
NFC, being embedded in a graphical user-interface, facilitates not only quick and easy creation of networks but 
also the analysis of e.g. earliest arrival time functions and inflow functions in a dynamic
equilibrium, which can be computed with a single click for further research. The flow models used are due to [[1]](#references) in 
the basic case without spillback and due to [[2]](#references) in the case of spillback. The algorithms to compute thin flows are 
based on [[2]](#references) and [[3]](#references).

At the moment, NFC is only able to handle single-commodity flow networks. For a useful tool to compute multi-commodity flows 
(although no equilibria), please check out [this repository](https://github.com/zimmer-m/multi-commodity-flows-over-time). 

NFC was developed being part of a bachelor thesis [[3]](#references) and then afterwards 
enhanced within the TU Berlin/ECMath project [*Dynamic Models and Algorithms for Equilibria in Traffic Networks*](https://www.coga.tu-berlin.de/v_menue/projects/mi12/).

## Requirements
NFC was designed to run on Linux-based OS. If the packages below can be installed, then the tool should run on other OS as well.
- [Python](https://python.org/) >= 3.7
- PyQt5
- [matplotlib](https://matplotlib.org/) >= 3.1
- [numpy](https://numpy.org/) >= 1.17
- [networkx](https://networkx.github.io/) >= 2.4
- [SCIP](https://www.scipopt.org/) >= 7.0

## Usage
Run mainControl.py to open the GUI.


## References
- [1] Koch, Ronald, and Skutella, Martin. "Nash equilibria and the price of anarchy for flows over time." Theory of Computing Systems 49.1 (2011): 71-97.
- [2] Sering, Leon, and Vargas Koch, Laura. "Nash flows over time with spillback." Proceedings of the Thirtieth Annual ACM-SIAM Symposium on Discrete Algorithms. Society for Industrial and Applied Mathematics, 2019.
- [3] Zimmer, Max. "Nash Flows Over Time: Models and Computation." Bachelor Thesis at TU Berlin (2017)
