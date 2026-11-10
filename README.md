# Early Learner Outcomes: Multi-Task Neural Network

This project explores whether a learner's first two weeks of activity in an online course can help predict two outcomes: whether the learner earns a **certificate**, and their **final grade band** (D, C, B, or A). It is an educational prototype for early outreach, not an automated decision system.

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

These results come from a held-out test split of **255 rows**, about 20% of the 1,272-row modeling dataset. The split was made by row, so the same learner can appear in more than one split when they have several course rows. A learner-grouped split is a planned next step. These results describe this test split only and do not guarantee performance on a new population.

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

## LIME: what it does and its limits

LIME explains one prediction by creating perturbed copies of the learner's inputs, scoring them with the model, and fitting a simple weighted model around that point. The output is a set of feature contributions for that learner.

These contributions describe the local approximation. They are not causal effects or guaranteed interventions.

LIME has been unreliable in this project and must be revised before its explanations are relied on:

- In some cases it did not produce a usable explanation. The local fit (R²) was low, or the local prediction differed from the model's probability.
- For one feature, the sign of LIME's contribution disagreed with a direct test on the model, where changing that input changed the probability in the opposite direction.

The app shows a quality warning when R² is below 0.8 or the local prediction differs from the model by more than 0.1. These are screening thresholds, not guarantees. Planned work: revise the LIME setup, and compare it with Integrated Gradients, which uses the network's own gradients and needs no surrogate model.

## Code excerpts

The excerpts below are selected from the private implementation. They are not runnable on their own.

### 1. Early-activity window

This selects each learner's activity from their first recorded day through 14 days later. It is the core of the feature design.

```python
# Find each learner's first recorded activity date.
events["date"] = pd.to_datetime(events["date"])
first_day = events.groupby("user_uuid", as_index=False)["date"].min()
first_day = first_day.rename(columns={"date": "first_event_date"})
events = events.merge(first_day, on="user_uuid", how="left")

# Keep only activity on or before 14 days after that first date (inclusive).
events["cutoff_date"] = events["first_event_date"] + pd.Timedelta(days=14)
early = events.loc[events["date"] <= events["cutoff_date"]].copy()
```

### 2. Shared network with two outputs

One dense layer learns a representation used by both tasks. Each output head is trained with its own loss.

```python
inputs = keras.Input(shape=(input_dim,), name="input")

# Shared representation used by both tasks; dropout reduces overfitting.
shared = keras.layers.Dense(128, activation="relu")(inputs)
shared = keras.layers.Dropout(0.5)(shared)

# Grade head: four class probabilities (D, C, B, A).
grade_output = keras.layers.Dense(4, activation="softmax", name="grade_output")(shared)
# Certificate head: a single probability.
cert_output = keras.layers.Dense(1, activation="sigmoid", name="cert_output")(shared)

model = keras.Model(inputs=inputs, outputs=[grade_output, cert_output])
```

### 3. LIME quality check

LIME's local fit is checked before its explanation is shown. This is the mechanism that flags explanations as unreliable.

```python
@staticmethod
def _add_explanation_quality(result: dict[str, Any], probability: float) -> None:
    # Compare the local surrogate's prediction with the model's own probability.
    error = abs(result["local_surrogate_prediction"] - probability)
    result["surrogate_absolute_error"] = float(error)

    # Flag the explanation when the surrogate fits poorly (R² < 0.8) or
    # disagrees with the model by more than 0.1.
    if result["local_fidelity_r2"] < 0.8 or error > 0.1:
        result["quality_warning"] = (
            "The local approximation may be unreliable: R² is below 0.8 or the "
            "surrogate differs from the model probability by more than 0.1."
        )
    else:
        result["quality_warning"] = None
```

### 4. Upload and prediction flow in the UI

The Streamlit app validates both uploads, then passes them to the shared inference service. The optional LIME explanation is requested with a learner ID.

```python
# Validate both CSVs and enforce the two-learner limit.
learners, activity = load_uploads(
    learners_file.getvalue(), activity_file.getvalue(), max_learners=MAX_LEARNERS
)

# Run the model and, if an ID was entered, a LIME explanation for that learner.
result = service().predict(learners, activity, explain_id or None)

# A quality warning is shown next to any LIME explanation.
if explanation["quality_warning"]:
    st.warning(explanation["quality_warning"])
```

## Technologies and boundaries

- TensorFlow/Keras, pandas, scikit-learn, LIME, Streamlit, Docker, and Azure Container Apps.
- Not included: customer data, learner-level records, engineered learner data, customer identifiers, trained model files, preprocessor files, secrets, and the full implementation.

This work is an educational prototype, not an automated decision or intervention system.
