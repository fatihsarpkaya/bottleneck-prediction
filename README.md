This repository contains artifacts for the ML-based bottleneck type prediction project. In this project, the sender predicts the AQM type at the bottleneck and determines whether it is FIFO or PIE.

This repository includes:

 - FABRIC testbed notebooks and descriptions for conducting experiments and collecting training and test data.
 - Google Colab notebook for training and evaluating the data and generating figures for the report.

Experiments are on the FABRIC testbed and to run this experiment on [FABRIC](https://fabric-testbed.net), you should have a FABRIC account with keys configured, and be part of a FABRIC project. You will need to have set up SSH keys and understand how to use the Jupyter interface in FABRIC.

# Data Collection

To reproduce our experiments and collect data for ML, log in to FABRIC's JupyterHub environment. Open a terminal from the launcher and run:

> git clone https://github.com/fatihsarpkaya/bottleneck-prediction.git

In the `main-training.ipynb` notebook, you can conduct experiments to collect training data. Similarly, the `main-evaluation.ipynb` notebook is for experiments to collect test data. These notebooks generate CSV files containing time-series data on congestion window (CWND), round-trip-time (RTT) and time-stamp. You can use these files for data processing in the following section.


# Training and Evaluation (Reproducing the Figures in the Report)

To train and evaluate the collected data, use the following Colab notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qR_eJaG1HieybAngOrT312PG8gI9AxSR?usp=sharing)

This Google Colab notebook allows you to download our experimental data from Google Drive, train and evaluate the ML models, and plot figures for the report. You can also modify the notebook to use your own experimental data and generate custom plots.


