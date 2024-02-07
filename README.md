# Computing Nash Flows Over Time and Thin Flows

The *Nash Flow Computation Tool* (NFC) allows the computation of thin flows with resetting and consequently Nash flows over time with or without spillback. 
NFC, being embedded in a graphical user-interface, facilitates not only quick and easy creation of networks but also the analysis of e.g. earliest arrival time functions and inflow functions in a dynamic equilibrium, which can be computed with a single click for further research. The flow models used are due to [[1]](#references) in the basic case without spillback and due to [[2]](#references) in the case of spillback. The algorithms to compute thin flows are based on [[2]](#references) and [[3]](#references). A complete guide on Nash flow over time can be found in [[4]](#references).

At the moment, NFC is only able to handle single-commodity flow networks. For a useful tool to compute multi-commodity flows (although no equilibria), please check out [this repository](https://github.com/LeonSering/multi-commodity-flows-over-time). 

NFC was developed being part of a bachelor thesis [[3]](#references) (found [here](/documentation/thesis.pdf)) and then afterwards enhanced within the TU Berlin/ECMath project *Dynamic Models and Algorithms for Equilibria in Traffic Networks*.

## Requirements

NFC was designed to run on Linux-based OS.

- [Python3](https://python.org/) >= 3.7
- [PyQt5](https://pypi.org/project/PyQt5/) >= 5.15
- [matplotlib](https://matplotlib.org/) >= 3.1
- [numpy](https://numpy.org/) >= 1.17
- [networkx](https://networkx.github.io/) >= 2.4
- [SCIP](https://www.scipopt.org/) >= 7.0

## Install (Windows / Mac)

For Windows or Mac we recommend to use a virtual machine [VirtualBox](https://www.virtualbox.org) running [Ubuntu 20.04.4 Focal Fossa](https://www.osboxes.org/ubuntu/).

## Install (Ubuntu)

**Warning:** Newer Ubuntu version might not have Qt5 anymore and it can be tricky to install it. However, [this](https://stackoverflow.com/questions/74110674/problem-installing-qt5-default-on-ubuntu-22-04#:~:text=1%20Answer&text=Ubuntu%2022.04%20repository%20dont%20have,by%20installing%20the%20qtbase5-dev.) might help. Otherwise, it might be easier to set up a virtual machine with VirtualBox and follow the steps below within the VM.

The following is tested in a virtual machine with VirtualBox 7.0 running Ubuntu 20.04.4 Focal Fossa 64bit.

Open a terminal and follow these steps:

1) Install **python3** (often already installed) and **git**:

   ```bash
   sudo apt update && sudo apt install python3 git
   ```

2. Install python packages:

   ```bash
   sudo apt install python3-pyqt5 python3-matplotlib python3-numpy python3-networkx
   ```

3. Clone the repository into a folder **NashFlowComputation**:

   ```bash
   git clone https://github.com/LeonSering/NashFlowComputation.git
   ```

4. Download **SCIP** from the [SCIP download page](https://www.scipopt.org/index.php#download). Choose the newest Debian package (ends with .deb). Tested with [SCIPOptSuite-8.1.0-Linux-ubuntu.deb](https://www.scipopt.org/download.php?fname=SCIPOptSuite-8.1.0-Linux-ubuntu.deb).

5. Install **SCIP** by either clicking on **SCIPOptSuite-8.1.0-Linux-ubuntu.deb** or by
   
   ```bash
   sudo apt install ./Downloads/SCIPOptSuite-8.1.0-Linux-ubuntu.deb 
   ```

6. Go into the **NashFlowComputation** directory with:

   ```bash
   cd NashFlowComputation
   ```

7. Open the GUI with:

   ```bash
   python3 mainControl.py
   ```

7. Set the SCIP path by clickling "Select binary" and choose the scip binary which can often be found under /usr/bin/scip (otherwise you can find it by using the command ```whereis scip```).

## Usage

#### Create the network and compute a Nash Flow Over Time

1. First choose if you want to compute a Nash Flow Over Time with Spillback (choose "Spillback") or without (choose "General").

2. Create a graph by clicking on an existing node and drag the mouse to create an arc. At the end all nodes (in particular the sink t) must be reachable from the source s.

3. Click on an edge to set the capacity and the transit times (and inflow and storage capacity for spillback networks). Don't forget to click "Update edge".

4. Set the network inflow rate.

5. If the network is created, click on "Compute Nashflow" to start the computation.

#### Optional settings

- You can rename and re-position nodes by clicking on them and editing the fields. Click "Update node" afterwards.

- You can save and load graphs via the "File"-menu.

- For no-spillback graph you can choose between three different algorithms. All should have the same result but the running time might vary drastically.

- To store the output for each thin flow into a file, choose a Output directory and deselect the Clean-up checkbox.

#### Inspect the Nash Flow Over Time

- Click on a node in the upper frame to inspect the earliest arrival time of that node.

- Click on an edge in the upper frame to inspect the cumulative in- and outflow the queue size and the edge load. Above the graphs the current queue is visualized.

- You can hide graphs by clicking on the respective line in the legend.

- Click on the intervals to see the Thin Flow of that interval.

- It is possible to run an animation by clicking the play button.

- You can also set a specific time. For example if you set the time to 10:
  
  - The upper frame shows the current flow state at snapshot time 10. 
  
  - The thin flow in the lower frame shows that thin flow for the particle/agent that starts at time 10.
  
  - By default active edges with no thin flow are displayed. You can show them by disabling the "Show edges w/out flow"-checkbox.

- The graph can be exported to pdf or pgf.

- The flow animation can be exported to an mp4-file by clicking on the black record button. It is possible to select the time window for the animation under the Option menu.

#### Compute only a single thin flow

- Start the ThinFlowComputation tool by running 
  
  ```bash
  python3 thinFlow_mainControl.py
  ```
  
  or by selecting "Open ThinFlowComputation" in the Options menu in the Nash Flow Computation Window.

- It is also possible to move the network from the NashFlowComputation window to the ThinFlowComputation by choosing "Move current graph to ThinFlowComputation" in the Options menu.

- Create the network in the upper canvas by drag and drop.

- Set the Capacity and the Flow bound (only for spillback thin flows) for each edge (transit times do not play any role for thin flows as long as you only include active edges). Don't forget to press "Update edge".

- Set the status (active or resetting) of each edge. (Note that resetting edges are always active. However for the program an resetting edge that is not active is assumed to be non-active and ignored.)

- The graph of active edges must by acyclic.

- Again there are three different algorithms for computing a non-spillback thin flow. All of them should output the same but my vary in the running time.

- Set the inflow rate.

- Click on "Compute Thin Flow" if the network is ready.

- The bottom canvas shows the computed thin flow. Next to each node is the $\ell'$ value and next to each node the $x'$ value.

## References

- [1] Koch, Ronald, and Skutella, Martin. "Nash equilibria and the price of anarchy for flows over time." Theory of Computing Systems 49.1 (2011): 71-97. [link](https://doi.org/10.1007/s00224-010-9299-y)
- [2] Sering, Leon, and Vargas Koch, Laura. "Nash flows over time with spillback." Proceedings of the Thirtieth Annual ACM-SIAM Symposium on Discrete Algorithms. Society for Industrial and Applied Mathematics, 2019. [link](https://doi.org/10.1137/1.9781611975482.57)
- [3] Zimmer, Max. "Nash Flows Over Time: Models and Computation." Bachelor Thesis at TU Berlin (2017) [link](/documentation/thesis.pdf)
- [4] Sering, Leon. "Nash flows over time." Dissertation (2020) [link](https://doi.org/10.14279/depositonce-10640)
