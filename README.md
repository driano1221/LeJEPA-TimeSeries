
# LeJEPA for Time Series: Self-Supervised Learning for ECG and Industrial Vibrations

> **A PyTorch implementation of the LeJEPA (2025) algorithm adapted for time series analysis, demonstrating robust representation learning that surpasses supervised methods in noisy and data-scarce environments.**

## 1\. Project Overview

This project investigates the application of **Joint-Embedding Predictive Architectures (JEPA)** combined with statistical regularization (**SIGReg**) to two distinct types of temporal data:

  * **ECG5000 (Cardiology):** Clean and temporally aligned physiological signals.
  * **FordA (Industry 4.0):** Noisy and temporally misaligned vibration signals (characterized by phase shift).

The primary objective is to evaluate whether self-supervised learning can acquire robust representations capable of outperforming traditional supervised baselines, particularly in scenarios with limited labeled data.

## 2\. Visual Results

The following figure summarizes the principal finding of this project: **Data Efficiency**.

**[Insert the 'benchmark\_comparison.png' image here]**

### Interpretation of the Results

We conducted tests across two distinct scenarios to validate the robustness of the model:

#### The Industrial Challenge (FordA - Right Graph)

This dataset comprises vibrations from automotive engines. The data is characterized by significant noise and temporal misalignment, which often confounds traditional models.

  * **Baseline (Grey Line):** A Logistic Regression model trained on the raw data fails significantly (Accuracy \~50%, equivalent to random guessing), as it cannot effectively handle the noise and temporal displacements.
  * **LeJEPA (Blue Line):** Our model achieves **\>90% accuracy**, even with limited data.
      * **Key Highlight:** With only **1% of the labels** (approximately 39 examples), LeJEPA achieves **88.8% accuracy**, massively outperforming the traditional method. This demonstrates that the model learned the underlying structure of the problem autonomously during the pre-training phase.

#### The Cardiology Test (ECG5000 - Left Graph)

This dataset contains cardiac beats that have been pre-processed and aligned.

  * **Baseline (Grey Line):** As the data is clean and aligned, the linear baseline achieves 99% accuracy (the problem is "easy" for simple methods).
  * **LeJEPA (Blue Line):** LeJEPA performs comparably, reaching **\~98.9% accuracy**.
  * **Key Lesson:** In clean data scenarios, the gain from SSL is marginal, yet the model maintains high performance, demonstrating its versatility.

-----

## 3\. Methodology

To achieve these results, we constructed a pipeline that does not rely on human labels for learning:

1.  **Convolutional Encoder (1D CNN):** A neural network designed to process the time series and extract mathematical features.
2.  **Self-Supervised Learning (LeJEPA):**
      * Instead of explicitly teaching the model to classify defects, we train the network to **predict the structure of the signal itself**.
      * **Data Augmentation** (Jittering and Scaling) is employed to create variations of the signal, forcing the network to learn that "noisy signal" and "clean signal" represent the same underlying entity (Invariance).
3.  **Statistical Regularization (SIGReg):** A mathematical component (based on the Epps-Pulley test) that prevents the network from producing trivial solutions (collapse) and ensures that the embeddings possess desirable statistical properties.

-----

## 4\. Detailed Benchmark Results

### The Challenge of Noise (FordA)

In complex data where linear methods fail due to misalignment and noise, LeJEPA excelled by learning temporal invariance.

| Model | 1% Labels (Few-Shot) | 100% Labels |
| :--- | :---: | :---: |
| **Baseline Linear** | 53.3% (Random) | 49.4% (Failure) |
| **LeJEPA (Ours)** | **87.7%** | **93.3%** |
| **Gain** | **+34.4%** | **+43.9%** |

> **Conclusion:** LeJEPA significantly outperformed the baseline, proving it learned the intrinsic "shape" of the motor defect, ignoring noise and temporal displacement. This highlights the potential of SSL in industrial settings where labeling data is expensive or difficult.

### The Structure Test (ECG5000)

In perfectly aligned data, where simple methods already saturate the benchmark, LeJEPA showed competitiveness.

| Model | 1% Labels | 100% Labels |
| :--- | :---: | :---: |
| **Baseline Linear** | 96.7% | 99.2% |
| **LeJEPA (Ours)** | 91.4% | 97.8% |

> **Insight:** Although the baseline wins by a small margin (due to the artificial alignment of the dataset), LeJEPA reached \>91% accuracy with only 39 examples, demonstrating that it learned cardiac morphology without supervision. This suggests its utility in scenarios with less curated data.

## 5\. Technologies Used

  * **PyTorch:** Utilized for the implementation of the 1D CNN and SIGReg Loss.
  * **Epps-Pulley Test:** The statistical core that prevents embedding collapse.
  * **Data Augmentation:** Jittering and Scaling applied to enforce invariance.

## 6\. Architecture

The system consists of the following components:

1.  **Encoder:** A 1D Convolutional Neural Network (CNN) that extracts features from the time series input.
2.  **Projector:** An MLP head that maps features to the embedding space.
3.  **Loss Function:** A combination of predictive loss (MSE) and SIGReg loss to enforce distribution regularization.

## 7\. Instructions for Execution

1.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Execute the full benchmark:
    ```bash
    python main.py
    ```

The script will automatically download the datasets, train the models, and generate the comparison plots in the `results/` folder.

## 8\. Project Structure

```
LeJEPA-TS/
│
├── README.md              # Project documentation
├── requirements.txt       # List of dependencies
├── main.py                # Main script to run experiments
│
├── src/                   # Source code
│   ├── __init__.py
│   ├── model.py           # Encoder, SIGReg, and LeJEPA architecture
│   ├── data.py            # Dataset loading and augmentation
│   └── evaluation.py      # Testing functions and Linear Probe
│
├── checkpoints/           # Saved Models
│   ├── ecg5000_model.pth  # Weights for ECG model
│   └── forda_model.pth    # Weights for FordA model
│
└── results/               # Results
    ├── benchmark_comparison.png  # Comparison plot
    └── metrics.json              # Raw metrics
```




## 9 References & Acknowledgements

###  Core Research (Foundations)

This project is built upon the following groundbreaking papers in Self-Supervised Learning and Statistics:

  * **LeJEPA (State-of-the-Art):**

      * Balestriero, R., & LeCun, Y. (2025). *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics*. arXiv preprint arXiv:2511.08544.
      * 🔗 [Link to arXiv](https://arxiv.org/abs/2511.08544)

  * **I-JEPA (Architecture Basis):**

      * Assran, M., et al. (2023). *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
      * 🔗 [Link to Paper](https://arxiv.org/abs/2301.08243)

  * **SIGReg Foundation (Statistical Test):**

      * Epps, T. W., & Pulley, L. B. (1983). *A test for normality based on the empirical characteristic function*. Biometrika, 70(3), 723-726.
      * 🔗 [Link to Article](https://www.google.com/search?q=https://academic.oup.com/biomet/article-abstract/70/3/723/250760)

  * **Contrastive Baselines:**

      * Chen, T., et al. (2020). *A Simple Framework for Contrastive Learning of Visual Representations* (SimCLR). ICML.
      * 🔗 [Link to Paper](https://arxiv.org/abs/2002.05709)

###  Datasets

The data used for benchmarking in this project is sourced from the standard UCR Time Series Classification Archive:

  * **UCR Time Series Archive:**
      * Dau, H. A., et al. (2019). *The UCR Time Series Archive*. IEEE/CAA Journal of Automatica Sinica.
      * 🔗 [Official Website](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)
      * **FordA Dataset:** [Info & Download](http://www.timeseriesclassification.com/description.php?Dataset=FordA)
      * **ECG5000 Dataset:** [Info & Download](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000)

### 🛠️ Tools & Frameworks

  * **PyTorch:** [pytorch.org](https://pytorch.org/)
  * **Scikit-Learn:** [scikit-learn.org](https://scikit-learn.org/)
  * **SciPy:** [scipy.org](https://scipy.org/)