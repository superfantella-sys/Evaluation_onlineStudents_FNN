# Early Learner Outcomes: Multi-Task Neural Network

This project explores whether a learner's first two weeks of activity in an online course can help predict two outcomes: whether the learner earns a **certificate**, and their **final grade band** (D, C, B, or A). It is an educational tool for early outreach, not an automated decision system.

This repository is a **portfolio walkthrough**. It contains selected, commented code excerpts, the held-out evaluation results, and a link to a hosted demo. The full implementation, data, trained model, and preprocessor are kept private.

## Problem

Course staff want to know which learners may need support before the course ends. The model uses only early activity, so it can flag risk while there is still time to act.

## Architecture

- **Data:** verified learners with a known final grade. Daily activity is aggregated per learner from their first recorded activity date through 14 days later, inclusive (up to 15 calendar dates).
- **Model:** a TensorFlow/Keras network with a shared dense layer and two outputs: a four-class grade head (softmax) and a certificate head (sigmoid).
- **Inference:** a Streamlit interface that accepts a learner CSV and an activity CSV, returns predictions and a bias check, and can explain one learner with LIME.
- **Deployment:** a Docker image hosted on Azure Container Apps, configured to scale to zero when idle.
- **Stack:** Python, TensorFlow, pandas, scikit-learn, LIME, Streamlit, Docker, Azure Container Apps.

## Run the demo

Open the hosted app:

https://ca-hvx-public.blackmoss-ec053946.switzerlandnorth.azurecontainerapps.io/

1. Upload a learner CSV and an activity CSV. The app's template download buttons show the required columns.
2. Click **Predict** to see certificate probabilities and grade bands.
3. Optionally enter a `user_uuid` from your learner file to request a LIME explanation for that learner.

Do not upload personal or sensitive data. The app accepts at most two learners per request.

The model and preprocessor are not published, so the demo cannot be run locally from this repository.

## Evaluation results

These results come from a held-out test split of **255 rows**, about 20% of the 1,272-row modeling dataset. The split was made by row, so the same learner can appear in more than one split when they have several course rows. A learner-grouped split is a planned next step. These results describe this test split only. Performance on other courses or learner groups has not been measured.

Certificate prediction, at a probability threshold of 0.5:

| Certificate metric | Result |
| --- | ---: |
| Accuracy | 91.8% |
| ROC AUC | 0.951 |
| Actually certified, predicted certified | 81 of 81 |
| Actually not certified, predicted certified | 21 of 174 |

Grade-band prediction:

| Grade metric | Result |
| --- | ---: |
| Overall grade-band accuracy | 62.7% |
| D recall | 138 of 149 |
| C recall | 11 of 57 |
| B recall | 3 of 6 |
| A recall | 8 of 43 |

Grade-band performance varies considerably by class. Of 54 learners predicted as band B, only 3 were correct.

## Framework comparison: TensorFlow vs PyTorch

To check whether the framework changes performance, the same network, data split, features, and training rules were rebuilt in PyTorch. Both used the same 255-row held-out test split. The PyTorch version ran on Azure Machine Learning with a conda environment pinned to `torch==2.14.1` and `pyarrow` for reading the Parquet inputs.

| Metric (test, 255 rows) | TensorFlow | PyTorch (seed 42) |
| --- | ---: | ---: |
| Certificate accuracy | 91.8% | 91.8% |
| Certificate ROC AUC | 0.951 | 0.953 |
| Grade-band accuracy | 62.7% | 63.5% |
| Grade-band log loss | 0.723 | 0.722 |

The framework does not change performance in a meaningful way. The differences are within the variation from changing the random seed: across five PyTorch seeds, test certificate AUC ranged from 0.944 to 0.953 and grade accuracy from 62.0% to 65.5%. Only a single split was used, so these results should be read as a comparison, not a guarantee.

## LIME: what it does and its limits

LIME explains one prediction by creating perturbed copies of the learner's inputs, scoring them with the model, and fitting a simple weighted model around that point. The output is a set of feature contributions for that learner.

These contributions describe the local approximation. They are not causal effects or guaranteed interventions.

LIME has been unreliable in this project and must be revised before its explanations are relied on:

- In some cases it did not produce a usable explanation. The local fit (R²) was low, or the local prediction differed from the model's probability.
- For one feature, the sign of LIME's contribution disagreed with a direct test on the model, where changing that input changed the probability in the opposite direction.

The app shows a quality warning when R² is below 0.8 or the local prediction differs from the model by more than 0.1. These are screening thresholds, not guarantees. Planned work: revise the LIME setup, and compare it with Integrated Gradients, which uses the network's own gradients and needs no surrogate model.

## Code excerpts

The files in [`code/`](code/) are selected from the private implementation, with comments explaining each part. They are not runnable on their own.

- [`01_early_activity_window.py`](code/01_early_activity_window.py): selects each learner's activity from their first recorded day through 14 days later.
- [`02_shared_two_head_network.py`](code/02_shared_two_head_network.py): one shared layer feeding the grade and certificate heads.
- [`03_lime_quality_check.py`](code/03_lime_quality_check.py): checks LIME's local fit and flags unreliable explanations.
- [`04_ui_prediction_flow.py`](code/04_ui_prediction_flow.py): the Streamlit upload, prediction, and LIME warning flow.

## Technologies and boundaries

- TensorFlow/Keras, pandas, scikit-learn, LIME, Streamlit, Docker, and Azure Container Apps.
- Not included: customer data, learner-level records, engineered learner data, customer identifiers, trained model files, preprocessor files, secrets, and the full implementation.

This work is educational, not an automated decision or intervention system.
