# Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow

Repository ini berisi **reproduksi kode + ringkasan + penjelasan teori per chapter** dari buku *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (O’Reilly)*.

## Tujuan
- Memperdalam pemahaman konsep Machine Learning & Deep Learning melalui implementasi langsung.
- Mendokumentasikan ringkasan dan teori yang dipakai pada setiap chapter, terhubung dengan kode di notebook.

## Struktur Repository
- `notebooks/` — notebook per chapter (kode + ringkasan + teori)
- `scripts/` — helper scripts (opsional)
- `assets/` — gambar/screenshot/hasil eksperimen (opsional)

> Catatan: Saya mengerjakan di **Google Colab** dan mengunggah notebook ke GitHub secara manual (tanpa push via bash).

## Cara Menjalankan (Google Colab)
1. Buka file `.ipynb` di folder `notebooks/`.
2. Jika perlu, install dependency:
    ```bash
    pip install -r requirements.txt
    ```
3. Jalankan cell dari atas ke bawah.

## Cara Menjalankan (Local — Opsional)
1. Buat environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # macOS/Linux
    # .venv\Scripts\activate   # Windows
    pip install -r requirements.txt
    ```
2. Jalankan Jupyter:
    ```bash
    jupyter lab
    ```

## Daftar Chapter
> Notebook akan tersedia setelah file di-upload dan di-commit di GitHub.

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
| 17 | Representation Learning and Generative Learning Using Autoencoders and GANs | `notebooks/ch17_autoencoders_gans.ipynb` |
| 18 | Reinforcement Learning | `notebooks/ch18_reinforcement_learning.ipynb` |
| 19 | Training and Deploying TensorFlow Models at Scale | `notebooks/ch19_training_deploying_tf_at_scale.ipynb` |

## Standar Isi Setiap Notebook (Wajib)
Setiap notebook minimal memuat:
1. **Ringkasan chapter** (bullet/paragraph singkat dan jelas)
2. **Reproduksi kode** (dijalankan sampai menghasilkan output)
3. **Penjelasan teori** (konsep, intuisi, rumus kunci bila relevan, serta kaitannya dengan kode)
4. **Eksperimen/Hasil** (metrik/grafik/evaluasi + interpretasi)
5. **Takeaways**
6. **Referensi**

## Catatan Integritas Akademik
- Notebook tidak hanya berisi hasil copy-paste; setiap chapter dilengkapi ringkasan dan penjelasan teori.
- Perubahan kecil (mis. versi library, path dataset, parameter) dicatat di notebook.

## Referensi Utama
- Aurélien Géron, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (O’Reilly).
