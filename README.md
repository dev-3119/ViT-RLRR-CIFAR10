# ViT-RLRR-CIFAR10

This repository contains the implementation of a parameter-efficient fine-tuning approach for Vision Transformers (ViTs) using the **Residual-based Low-Rank Rescaling (RLRR)** method on the CIFAR-10 dataset.

---

## 1. Objective

The goal of this project is to fine-tune a pre-trained Vision Transformer (ViT-B/16) using the RLRR method. This approach aims to achieve high classification accuracy on CIFAR-10 while updating only a minimal set of new parameters, thereby reducing computational overhead and preserving the generalization ability of the pre-trained model.

---

## 2. Paper Summary

**Title:** Low-Rank Rescaled Vision Transformer Fine-Tuning: A Residual Design Approach  
**Authors:** Wei Dong, Xing Zhang, Bihui Chen, Dawei Yan, Zhijun Lin, Qingsen Yan, Peng Wang, Yang Yang  
**Publication:** Q1 Journal, 2024  

**Summary:**  
The paper introduces the RLRR method as a novel parameter-efficient fine-tuning (PEFT) technique for Vision Transformers. Instead of fine-tuning all parameters, RLRR leverages low-rank trainable matrices to rescale the frozen pre-trained weights via a residual connection. This design maintains the inherent generalization of the ViT model while allowing it to adapt effectively to downstream tasks. Experimental results demonstrate that RLRR achieves competitive accuracy with significantly fewer trainable parameters compared to methods like LoRA and AdaptFormer.

---

## 3. Implementation Approach

### Model Architecture
1. **Pre-Trained ViT Backbone:**  
   - The project uses the ViT-B/16 model from TensorFlow Hub as a frozen feature extractor.
   
2. **RLRR Adaptation Layer:**  
   - A custom RLRR layer is added to rescale the frozen weights. It introduces trainable low-rank matrices and a residual connection to adapt the model to CIFAR-10.
   
3. **Classification Head:**  
   - A final dense layer with 10 output neurons (for the 10 CIFAR-10 classes) and softmax activation produces the class predictions.

### Data Preprocessing
- **Dataset:** CIFAR-10 (60,000 images, 10 classes)
- **Image Preprocessing:**  
  - Images are resized from 32×32 to 224×224 pixels.
  - Pixel values are normalized to the range [0, 1].
- **Data Pipeline:**  
  - Implemented using `tf.data.Dataset` with efficient batching, mapping, and prefetching to prevent out-of-memory (OOM) issues.

### Training & Evaluation
- **Training:**  
  - The model is trained for 5 epochs using the Adam optimizer and categorical crossentropy loss.
  - Only the RLRR layer and the classification head are trainable, while the ViT backbone remains frozen.
- **Evaluation:**  
  - Test accuracy, loss, confusion matrix, and classification report are computed to assess model performance.
- **Visualization:**  
  - Plots such as training vs. validation loss, ROC curves, and confusion matrices are generated to analyze the model's behavior.

---

## 4. Repository Structure

```
ViT-RLRR-CIFAR10/
│
├── README.md               # This file: Paper summary, implementation approach, results, etc.
├── data/                   # Dataset links & instructions to acquire CIFAR-10
├── src/                    # Source code for model training, evaluation, and utility functions
├── notebooks/              # Jupyter Notebooks for experiments and visualizations
└── results/                # Performance metrics, graphs (loss curves, ROC, confusion matrix), and analysis reports
```

---

## 5. How to Run

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- TensorFlow Hub
- Other libraries: NumPy, Matplotlib, Seaborn, scikit-learn

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/ViT-RLRR-CIFAR10.git
   cd ViT-RLRR-CIFAR10
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Training the Model
Run the training script in the `src/` folder:
```bash
python src/train.py
```

### Evaluating the Model
After training, you can run:
```bash
python src/evaluate.py
```
This script will generate the classification report, confusion matrix, and ROC curves.

---

## 6. Observations and Results

- **Test Accuracy:** ~96%
- **Training vs. Validation Loss:** The loss curves indicate consistent convergence with minimal overfitting.
- **Confusion Matrix & Classification Report:** The results show high precision and recall across CIFAR-10 classes.
- **ROC Curves:** AUC values close to 1 for all classes, demonstrating strong model performance.

---

## 7. Conclusion

The **ViT-RLRR fine-tuning method** efficiently adapts a pre-trained Vision Transformer to the CIFAR-10 dataset while maintaining most of the original model's knowledge. This parameter-efficient approach not only reduces computational costs but also achieves high accuracy, making it ideal for real-world applications with limited resources.
