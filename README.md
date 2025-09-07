# Integrating BERT with CNN and BiLSTM for Explainable Depression Detection

This repository contains a PyTorch implementation replicating the research paper **"Integrating Bert With CNN and BILSTM for Explainable Detection of Depression in Social Media Contents"**. The project focuses on building and evaluating advanced NLP models for depression detection and leverages Explainable AI (XAI) to interpret their decisions.

## About The Project

The primary goal of this project is to implement and validate the models proposed in the original research for detecting depression from social media text. This involves several key phases:
* **Data Preprocessing**: Cleaning and tokenizing text data from social media.
* **Model Construction**: Building three architectures: Fine-tuned BERT, BERT-BiLSTM, and BERT-CNN.
* **Model Training & Evaluation**: Training the models on a public dataset and evaluating their performance using standard metrics.
* **Explainable AI (XAI)**: Using model interpretation techniques to visualize and understand *why* a model makes a particular prediction, enhancing transparency and trust.

### Original Research Paper
> C. Xin and L. Q. Zakaria, "Integrating Bert With CNN and BILSTM for Explainable Detection of Depression in Social Media Contents," in *IEEE Access*, vol. 12, pp. 161203-161212, 2024, doi: 10.1109/ACCESS.2024.3488081.

---

## Project Structure

This project is organized into several Jupyter notebooks, each representing a stage of the replication process.

* **`Integrating_Bert_With_CNN_and_BiLSTM_for_Explainable_Detection.ipynb`**
    * Covers the initial steps of the project.
    * Data loading, cleaning, and preprocessing.
    * Definition of the `FineTunedBERT` model architecture.
    * The complete training loop for the `FineTunedBERT` model.

* **`Integrating_Bert_With_CNN_and_BiLSTM_for_Explainable_Detection_v1.ipynb`**
    * Focuses on the evaluation and interpretation of the first model.
    * Loading the saved `FineTunedBERT` model.
    * Evaluating the model on the test set to get metrics like Accuracy, Precision, Recall, and F1-Score.
    * Implementing the Explainable AI (XAI) part using the `transformers-interpret` library to visualize word importance.

* **`Integrating_Bert_With_CNN_and_BiLSTM_for_Explainable_Detection_v2.ipynb`**
    * Builds upon the previous steps by implementing the second model.
    * Definition of the `BERT+BiLSTM` model architecture.
    * Training, evaluation, and explainability analysis for the `BERT+BiLSTM` model.

---

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

This project uses Python and several key data science libraries. You can install them using pip.
```sh
pip install torch transformers scikit-learn pandas transformers-interpret
```

### Installation & Setup

1.  **Clone the repo**
    ```sh
    git clone [https://github.com/mrranger939/Integrating_Bert_With_CNN_and_BiLSTM_for_Explainable_Detection.git](https://github.com/mrranger939/Integrating_Bert_With_CNN_and_BiLSTM_for_Explainable_Detection.git)
    ```
2.  **Download the Dataset**
    [cite_start]The study uses the **Depression Reddit Dataset (Cleaned)** from Kaggle[cite: 121, 427].
    * Download it here: [https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned](https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned)
    * Create a `data` directory in the project root and place the `depression_dataset_reddit_cleaned.csv` file inside it.

3.  **Run the Notebooks**
    It is highly recommended to run these notebooks in an environment with GPU access, such as Google Colab. Open and run the notebooks sequentially to follow the project's workflow.

---

## Results and Screenshots

This section documents the results obtained during the replication and includes visualizations from the Explainable AI analysis.

### Performance on Reddit Dataset

The performance of the replicated **Fine-tuned BERT** model is shown below, alongside the results reported in the original paper (Table 2). The close alignment of the scores validates the correctness of the implementation.

| Metric | Replicated Result | Paper's Result  |
| :--- | :---: | :---: |
| **Accuracy** | 0.9845 | 0.981 |
| **Precision**| 0.9856 | 1.0 |
| **Recall** | 0.9831 | 0.979 |
| **F1-Score** | 0.9843 | 0.981 |

### Explainability Visualizations

*(You can add your screenshots here. Simply drag and drop the images into this section in the GitHub editor.)*

**Example 1: Correctly identifying a depressive post**
> 
<img width="719" height="69" alt="download" src="https://github.com/user-attachments/assets/50ab0063-dc5d-401f-ba96-2c957619864e" />


**Example 2: Correctly identifying a non-depressive post**
<img width="805" height="74" alt="download" src="https://github.com/user-attachments/assets/965efda9-cdf9-401a-98a2-ae1241451346" />
