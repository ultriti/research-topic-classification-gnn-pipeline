# Project Documentation: Graph Neural Network (GNN) Pipeline for Citation Classification

## 1. Project Document Introduction

### 1.1 Overview
This project delivers an end-to-end **Graph Neural Network (GNN) Pipeline** designed for automated node classification within citation networks. Developed in a PyTorch and PyTorch Geometric environment, the system classifies scientific research papers into predefined topic categories by exploiting both node-level textual feature representations and edge-level topological citation relationships.

Traditional Machine Learning approaches treat documents as isolated entities (e.g., using Bag-of-Words or TF-IDF representations). This pipeline demonstrates the transformative advantage of **Graph Convolutional Networks (GCN)**, which aggregate contextual features from neighboring papers in the citation graph to achieve significantly higher classification accuracy and macro F1 scores.

```
 +-----------------------------------------------------------------------------------+
 |                                   CORA GRAPH                                      |
 |                                                                                   |
 |  [Paper Node A] <---- (Citation Edge) ----> [Paper Node B]                       |
 |  Features: 1,433-dim binary word vector     Features: 1,433-dim binary word vector |
 +-----------------------------------------------------------------------------------+
                                           |
                                           v
 +-----------------------------------------------------------------------------------+
 |                           GNN PIPELINE ARCHITECTURE                               |
 |                                                                                   |
 |  1. Baseline Model : TF-IDF + Random Forest Classifier (Accuracy: 58.00%)         |
 |  2. Advanced GCN   : 2-Layer GCNConv + ReLU + Dropout (Accuracy: 78.20%)          |
 |  3. Export Engine  : PyTorch -> ONNX Serialization (Dynamic Graph Axes)           |
 +-----------------------------------------------------------------------------------+
```

### 1.2 Dataset Specification: Cora Citation Network
The pipeline is benchmarked on the gold-standard **Cora** dataset loaded via `torch_geometric.datasets.Planetoid`:

| Property | Description / Metric |
| :--- | :--- |
| **Dataset Name** | Cora (Planetoid Dataset Wrapper) |
| **Total Nodes (Papers)** | 2,708 papers |
| **Total Edges (Citations)** | 10,556 citation links |
| **Feature Dimensions** | 1,433 binary word features per node (indicating word presence/absence) |
| **Target Classes** | 7 topic categories (*Neural Networks, Probabilistic Methods, Genetic Algorithms, Theory, Case Based, Reinforcement Learning, Rule Learning*) |
| **Supervision Split** | **Train Mask**: 140 nodes (20 per class) <br> **Validation Mask**: 500 nodes <br> **Test Mask**: 1,000 nodes |

---

## 2. Use Cases

The Graph Neural Network pipeline addresses several high-value real-world application scenarios across academic, enterprise, and domain-specific knowledge graphs:

1. **Automated Academic & Scientific Literature Classification**
   - Automatically categorizes millions of unclassified preprints, research papers, and technical reports (e.g., on arXiv, IEEE, PubMed, Google Scholar) into granular domain hierarchies based on abstract/full-text keywords and citation graphs.

2. **Digital Library Recommendation Engines & Citation Analysis**
   - Enhances paper-to-paper recommendation systems by learning rich vector embeddings for research articles that incorporate structural citation context alongside semantic textual content.

3. **Enterprise Knowledge Graph Node Labeling**
   - Extends beyond academic text to enterprise knowledge graphs, such as mapping corporate entities, categorizing patents based on filing connections, or classifying wiki documents within internal knowledge bases.

4. **Social Network & User Profiling**
   - Applies the same node-classification architecture to social network graphs (e.g., predicting user interests or communities based on user profile text and follower/following interaction graphs).

5. **Fraud, Risk & Fraud Ring Detection**
   - Identifies suspicious entities in transaction graphs (e.g., financial fraud networks) where node features represent transaction histories and edges represent monetary interactions.

---

## 3. Industry Value & Business Impact

| Strategic Value Pillar | Impact & Benefit |
| :--- | :--- |
| **+20.2% Accuracy Boost Over Baseline** | By transitioning from isolated document classification (Random Forest: 58.00% accuracy) to relational graph learning (GCN: 78.20% accuracy), the system captures structural context that text alone misses. |
| **Massive Reduction in Data Labeling Costs** | The GCN model operates effectively in a **semi-supervised setting**, requiring labels for only 140 training nodes out of 2,708 papers (5.17% label coverage) while accurately generalizing across 1,000 test documents. |
| **Production-Ready Enterprise Serialization** | Fully integrated with **ONNX (Open Neural Network Exchange)** export routines, enabling seamless deployment to C++, Rust, Go, or edge microservices without requiring heavy PyTorch runtime overhead. |
| **Scalable Architectural Blueprint** | Serves as a modular foundation for complex enterprise GNN tasks including link prediction, graph classification, and dynamic graph embedding. |

---

## 4. Tech Stack Used & In-Depth Explanation

```
+---------------------------------------------------------------------------------------+
|                                  TECHNOLOGY STACK                                     |
+---------------------------------------------------------------------------------------+
| Runtime Environment   | Python 3.11.9                                                 |
| Graph Neural Network  | PyTorch Geometric (PyG), torch_geometric.nn.GCNConv          |
| Deep Learning Core    | PyTorch (torch, torch.nn, torch.optim)                        |
| Baseline ML Framework | Scikit-Learn (TfidfTransformer, RandomForestClassifier)      |
| Data Analytics        | Pandas, NumPy                                                 |
| Visualization         | Seaborn, Matplotlib, NetworkX                                 |
| Model Export & Interop| ONNX, ONNXScript (torch.onnx.export - Opset 18)                |
+---------------------------------------------------------------------------------------+
```

### 4.1 Technology Stack Details & Explanations

1. **Python 3.11.9**
   - The primary programming runtime, offering high-performance computation and seamless ecosystem compatibility with modern PyTorch Geometric libraries.

2. **PyTorch (`torch`)**
   - The underlying open-source deep learning framework providing tensor computation with GPU/CPU acceleration, dynamic computation graphs, automatic differentiation (`torch.autograd`), and modular neural network definitions (`torch.nn.Module`).

3. **PyTorch Geometric (`torch_geometric` / PyG)**
   - The flagship library built on PyTorch for deep learning on graph-structured data.
   - **`Planetoid`**: Benchmark loader used to import the Cora citation graph along with graph masks (`train_mask`, `val_mask`, `test_mask`).
   - **`GCNConv`**: Implementation of the Graph Convolutional operator from Kipf & Welling (2017). Performs neighborhood message passing according to the formula:
     $$\mathbf{X}^{(l+1)} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \mathbf{X}^{(l)} \mathbf{W}^{(l)}$$
     where $\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}_N$ is the adjacency matrix with added self-loops, and $\mathbf{\hat{D}}$ is its degree matrix.

4. **Scikit-Learn (`sklearn`)**
   - Used for constructing the traditional baseline machine learning pipeline.
   - **`TfidfTransformer`**: Converts raw binary word feature matrices into Term Frequency-Inverse Document Frequency (TF-IDF) weighted features.
   - **`RandomForestClassifier`**: Configured with 100 decision trees, square-root feature selection, and balanced class weights to train a text-only baseline model.
   - **`metrics` (`accuracy_score`, `f1_score`, `confusion_matrix`)**: Computes quantitative classification metrics and macro F1 scores.

5. **Pandas & NumPy**
   - High-performance tabular data structures (`pd.DataFrame`, `pd.Series`) and multi-dimensional array processing tools used for class distribution analysis and benchmark summary tables.

6. **Seaborn & Matplotlib**
   - Visualization libraries used to generate class balance bar charts and confusion matrix heatmaps to inspect true-positive vs false-positive predictions across all 7 topic classes.

7. **ONNX & ONNXScript (`torch.onnx`)**
   - **Open Neural Network Exchange**: Serializes PyTorch GNN models into a cross-platform format (`Artifacts/simple_gcn_cora.onnx`).
   - Configured with `opset_version=18` and `dynamic_axes` for dynamic node counts (`num_nodes`) and dynamic edge dimensions (`num_edges`), ensuring deployment flexibility in high-speed enterprise serving platforms (e.g., ONNX Runtime, Triton Inference Server).

---

## 5. Pipeline Architecture & Model Training Details

### 5.1 Baseline Model Architecture
* **Features**: TF-IDF Transformed 1,433 Bag-of-Words vectors.
* **Classifier**: `RandomForestClassifier(n_estimators=100, max_features='sqrt', class_weight='balanced', random_state=42)`
* **Test Performance**: Accuracy = **58.00%**, Macro F1 = **58.18%**.

### 5.2 GCN Model Architecture (`SimpleGCN`)
```python
class SimpleGCN(nn.Module):
    def __init__(self, input_dim=1433, hidden_dim=32, output_dim=7, dropout=0.5):
        super().__init__()
        self.convo1 = GCNConv(input_dim, hidden_dim)
        self.convo2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.convo1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convo2(x, edge_index)
        return x
```

### 5.3 Model Training Strategy & Regularization
* **Optimizer**: Adam (`learning_rate = 0.01`).
* **Loss Function**: Cross-Entropy Loss (`F.cross_entropy`) evaluated strictly over `train_mask`.
* **Early Stopping**: Monitored on `val_mask` accuracy with `patience = 20` epochs. Training converged and stopped early at **Epoch 33**.
* **Best Model Checkpointing**: State dictionary (`best_state`) restored prior to final test set evaluation.

---

## 6. Performance Evaluation & Model Comparison

| Metric | Baseline Model (TF-IDF + Random Forest) | Graph Neural Network (2-Layer GCN) | Relative Boost / Delta |
| :--- | :---: | :---: | :---: |
| **Model Type** | Text Feature Only (Tabular ML) | Graph Convolutional (Text + Network Structure) | **Hybrid Relational Learning** |
| **Accuracy** | **58.00%** (0.5800) | **78.20%** (0.7820) | **+20.20% Absolute Improvement** |
| **Macro F1 Score** | **58.18%** (0.5818) | **77.74%** (0.7774) | **+19.56% Absolute Improvement** |
| **Early Stopping** | N/A | Epoch 33 | Optimal Convergence |
| **Deployment Format** | Python Pickle / Joblib | **ONNX Format** (`simple_gcn_cora.onnx`) | Portable Enterprise Inference |

---

## 7. Model Export & Production Deployment

The trained GCN model is exported to an ONNX artifact with dynamic tensor dimensions:

```python
onnx_file_path = "Artifacts/simple_gcn_cora.onnx"

torch.onnx.export(
    model,
    (data.x, data.edge_index),
    onnx_file_path,
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=["node_features", "edge_indices"],
    output_names=["logits"],
    dynamic_axes={
        "node_features": {0: "num_nodes"},
        "edge_indices": {1: "num_edges"}
    },
)
```

### Artifact Files Location
* **Jupyter Notebook**: `model_training.ipynb`
* **Exported ONNX Model**: `Artifacts/simple_gcn_cora.onnx`
* **Raw & Processed Data**: `data/Planetoid/cora/`
