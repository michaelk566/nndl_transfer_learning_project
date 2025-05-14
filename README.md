# Transfer Learning Final Project — COMS 4995 (NNDL)

**The notebook is split into clear sections and is easy to follow, but we’ve included this README for additional guidance.**

From a high-level perspective, the notebook does the following:

1. **Loads your dataset**  
   * expects `train_images/`, `test_images/`, `train_data.csv`, `superclass_mapping.csv`, `subclass_mapping.csv`  
   * set `base_dir = "/path/to/your/folder/"` in the first code cell

2. **Exploratory stats** 
   
    Computes some key stats to help visualize the training dataset 

3. **Outlier Exposure**  
   * downloads Tiny-ImageNet  
   * samples 2000 images and labels them **novel**  
   * saves them into `train_images/novel_*.jpg`

4. **Model & training** – dual-head ResNet-50  
Call **`train_resnet50(**kwargs)`**. 

Key arguments:

| Param | Purpose | Default value |
|-------|---------|---------------|
| `use_novel_data` | Include Tiny-ImageNet outliers. | `True` |
| `use_augmentation` | Manual flips / crops / color jitter. | `False` |
| `use_nonlinear_head` | 512-unit MLP heads instead of linear. | `False` |
| `use_dropout` | Enable dropout inside heads. | `True` |
| `dropout_p` | Dropout probability. | `0.5` |
| `use_cosine_classifier` | Cosine-similarity logits. | `False` |
| `num_epochs` | Training epochs. | `20` |

The notebook prints **validation accuracy & cross-entropy** for both heads each epoch.

We varied all these flags in our project. Varying the params to match the experiments in our report will reproduce produce our results 

5. **Prediction & CSV export**  
   Runs on `test_images/`, applies confidence thresholds for novel detection if flag it set, and writes predictions to:

   - `results/test_predictions_softmax.csv`  - soft-max thresholds  
   - `results/test_predictions_logit.csv`    - raw-logit thresholds

---

**Note:** In the code blocks of the notebook we cite whenever we used AI to help generate boilerplate code. 

**Authors:**  
Michael Khanzadeh — <mmk2258@columbia.edu>  
Ryan Huang — <ry3129@columbia.edu>
