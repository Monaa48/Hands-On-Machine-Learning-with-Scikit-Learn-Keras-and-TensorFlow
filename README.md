# Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow

This repository contains **code reproductions, chapter summaries, and theoretical explanations** based on the book *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (O’Reilly) by Aurélien Géron.

## Objective
- Strengthen practical understanding of core **Machine Learning** and **Deep Learning** concepts through hands-on implementation.
- Document chapter-level **summaries + theory**, directly connected to the notebook code.

## Progress
✅ Completed: Chapter 01–19 (see notebook list below).

> Note: I work using **Google Colab** and upload notebooks to GitHub manually (no `git push` via bash).

## Repository Structure
- `notebooks/` — one notebook per chapter (code + summary + theory)
- `scripts/` — helper scripts (optional)
- `assets/` — images/snapshots/experiment outputs (optional)

## How to Run (Google Colab)
1. Open any `.ipynb` file inside `notebooks/`.
2. (Optional) Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run cells from top to bottom.

## How to Run (Local — Optional)
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   # .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```
2. Start Jupyter:
   ```bash
   jupyter lab
   ```

## Chapter Overview
- **01 — The Machine Learning Landscape:** what ML is, why it matters, main task types, key concepts, and common pitfalls.
- **02 — End-to-End ML Project:** full workflow (data → EDA → feature engineering → pipelines → training → tuning → evaluation).
- **03 — Classification:** binary/multiclass classification, metrics (precision/recall/F1), ROC/PR curves, confusion matrices, and error analysis.
- **04 — Training Models:** linear models, gradient descent variants, regularization, learning curves, and polynomial features.
- **05 — Support Vector Machines:** maximum-margin classifiers/regressors, kernels, soft margin, and practical tuning.
- **06 — Decision Trees:** splitting criteria, regularization, visualization, and interpretability.
- **07 — Ensemble Learning and Random Forests:** bagging, boosting, random forests, stacking, and bias–variance trade-offs.
- **08 — Dimensionality Reduction:** PCA, kernel PCA, LLE, and when/why to reduce dimensions.
- **09 — Unsupervised Learning:** clustering (k-means, DBSCAN), mixture models, anomaly detection basics, and evaluation without labels.
- **10 — Intro to Neural Networks with Keras:** MLP basics, training workflow, callbacks, and model evaluation.
- **11 — Training Deep Neural Networks:** initialization, normalization, optimization, regularization, and training stability.
- **12 — Custom Models & Training with TensorFlow:** custom layers/models, custom losses, GradientTape, and custom training loops.
- **13 — Data Loading & Preprocessing with TensorFlow:** `tf.data`, input pipelines, preprocessing layers, and efficient training input.
- **14 — Deep Computer Vision with CNNs:** CNN building blocks, modern architectures, transfer learning, and visualization.
- **15 — Sequences with RNNs & CNNs:** text/time-series pipelines, embeddings, RNN/LSTM/GRU, and sequence modeling patterns.
- **16 — NLP with RNNs & Attention:** attention intuition, seq2seq basics, and practical text modeling workflow.
- **17 — Autoencoders & GANs:** representation learning, VAEs, GAN training dynamics, and evaluation caveats.
- **18 — Reinforcement Learning:** Markov decision processes, policy/value methods, and deep RL intuition.
- **19 — Training & Deploying at Scale:** TF Serving, export formats, deployment patterns, and scaling considerations.

## Chapter List (use your exact filenames)
| No | Chapter | Notebook |
|---:|---|---|
| 01 | The Machine Learning Landscape | `notebooks/ch01_machine_learning_landscape.ipynb` |
| 02 | End-to-End Machine Learning Project | `notebooks/ch02_end_to_end_ml_project.ipynb` |
| 03 | Classification | `notebooks/ch03_classification.ipynb` |
| 04 | Training Models | `notebooks/ch04_training_models.ipynb` |
| 05 | Support Vector Machines | `notebooks/ch05_support_vector_machines.ipynb` |
| 06 | Decision Trees | `notebooks/ch06_decision_trees.ipynb` |
| 07 | Ensemble Learning and Random Forests | `notebooks/ch07_ensemble_learning_random_forests.ipynb` |
| 08 | Dimensionality Reduction | `notebooks/ch08_dimensionality_reduction.ipynb` |
| 09 | Unsupervised Learning Techniques | `notebooks/ch09_unsupervised_learning.ipynb` |
| 10 | Introduction to Artificial Neural Networks with Keras | `notebooks/ch10_ann_keras.ipynb` |
| 11 | Training Deep Neural Networks | `notebooks/ch11_training_deep_nns.ipynb` |
| 12 | Custom Models and Training with TensorFlow | `notebooks/ch12_custom_models_tf.ipynb` |
| 13 | Loading and Preprocessing Data with TensorFlow | `notebooks/ch13_data_loading_preprocessing_tf.ipynb` |
| 14 | Deep Computer Vision Using Convolutional Neural Networks | `notebooks/ch14_cnn_computer_vision.ipynb` |
| 15 | Processing Sequences Using RNNs and CNNs | `notebooks/ch15_sequences_rnns_cnns.ipynb` |
| 16 | Natural Language Processing with RNNs and Attention | `notebooks/ch16_nlp_rnns_attention.ipynb` |
| 17 | Representation Learning and Generative Learning Using Autoencoders and GANs | `notebooks/ch17_autoencoders_gans.ipynb
` |
| 18 | Reinforcement Learning | `notebooks/ch18_reinforcement_learning.ipynb` |
| 19 | Training and Deploying TensorFlow Models at Scale | `notebooks/ch19_training_deploying_tf_at_scale.ipynb` |

## Notebook Content Standard
Each notebook includes:
1. Chapter summary
2. Reproduced code (executed to produce outputs)
3. Theory explanation (intuition + key formulas when relevant)
4. Experiments/results (plots/metrics + interpretation)
5. Takeaways
6. References

## Academic Integrity
- Each chapter contains original explanation and interpretation, not only copy-pasted outputs.
- Any small changes (library versions, dataset paths, parameter adjustments) are noted inside the notebook.

## Primary Reference
- Aurélien Géron, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (O’Reilly).
