# Early Learner Outcomes: Multi-Task Neural Network

This capstone project explores whether a learner's early activity can help predict two course outcomes: **certification** and **final grade band**. It combines data preparation, a TensorFlow neural network, local explanations, and a prototype inference interface.

This repository is a **portfolio walkthrough**. It contains selected illustrative code excerpts, not a complete runnable application. The original data, trained model, preprocessor, and full implementation are not included.

## Approach

1. Select verified learners with a known final grade.
2. Aggregate daily activity by learner from the first recorded activity date through 14 days after it, inclusive. This covers up to 15 calendar dates.
3. Use a shared neural-network layer with two outputs: certificate probability and a four-class grade prediction.
4. Evaluate the model on a held-out test split and use LIME to explore individual certificate predictions.

Grade bands are D: below 0.20; C: 0.20 to below 0.70; B: 0.70 to below 0.90; and A: 0.90 or above.

### Selected code: early activity

The implementation finds a first activity date per learner and keeps activity through the inclusive cutoff. This excerpt omits the subsequent aggregation and merge.

```python
events["date"] = pd.to_datetime(events["date"])
first_day = events.groupby("user_uuid", as_index=False)["date"].min()
first_day = first_day.rename(columns={"date": "first_event_date"})
events = events.merge(first_day, on="user_uuid", how="left")
events["cutoff_date"] = events["first_event_date"] + pd.Timedelta(days=14)
early = events.loc[events["date"] <= events["cutoff_date"]].copy()
```

### Selected code: two output heads

The shared layer learns a representation used by both tasks. The grade head returns four class probabilities; the certificate head returns one probability.

```python
inputs = keras.Input(shape=(input_dim,), name="input")
shared = keras.layers.Dense(128, activation="relu")(inputs)
shared = keras.layers.Dropout(0.5)(shared)
grade_output = keras.layers.Dense(4, activation="softmax", name="grade_output")(shared)
cert_output = keras.layers.Dense(1, activation="sigmoid", name="cert_output")(shared)
model = keras.Model(inputs=inputs, outputs=[grade_output, cert_output])
```

### Inference prototype

A hosted Streamlit demo is available at https://ca-hvx-public.blackmoss-ec053946.switzerlandnorth.azurecontainerapps.io/. Users upload a learner CSV and a daily-activity CSV; the service prepares the features and returns a certificate probability and predicted grade band. A user can request a LIME explanation for one learner. The model stays on the server.

```python
# Illustrative excerpt: parse the uploads and run the shared inference service.
learners, activity = load_uploads(learner_bytes, activity_bytes)
service = get_service()
result = service.predict(learners, activity, explain_user_uuid)
```

The excerpt depends on helpers from the private implementation and is not runnable on its own. No public endpoint or executable demo is included in this repository.

## Held-out test results

The following results are from the retrained TensorFlow model on **255 held-out rows**: approximately 20% of the 1,272-row filtered modeling dataset. They describe this test split, not a guarantee of performance on a new population.

| Certificate metric | Result |
| --- | ---: |
| Accuracy | 91.8% |
| ROC AUC | 0.951 |
| Actual certified correctly predicted certified | 81 of 81 |
| Actual not certified incorrectly predicted certified | 21 of 174 |

These classification counts use a certificate probability threshold of 0.5.

| Grade metric | Result |
| --- | ---: |
| Overall grade-band accuracy | 62.7% |
| D recall | 138 of 149 |
| C recall | 11 of 57 |
| B recall | 3 of 6 |
| A recall | 8 of 43 |

Grade-band performance varied considerably across classes. In particular, 3 of 54 predictions of band B were correct.

The split was made by row, so the same learner could appear in more than one split when they had multiple course rows. A learner-grouped split is a worthwhile next evaluation.

## Local explanations with LIME

LIME approximates the model around one prediction by creating perturbed inputs, querying the model, and fitting a simpler weighted local model. Its feature contributions describe the approximation, not causal effects or guaranteed interventions.

The LIME results above are from a local check on training and held-out rows. LIME should be revised before it is relied on: in some scenarios it failed to provide a usable explanation, and its feature signs were inconsistent with direct tests on the model. Integrated Gradients is a candidate replacement for comparison.

## Technologies and boundaries

- TensorFlow/Keras, pandas, scikit-learn, and LIME.
- Streamlit for the hosted demo, packaged as a Docker image and run on Azure Container Apps.
- No customer data, learner-level records, model artifacts, or secrets are published here.

This work is an educational prototype, not an automated decision or intervention system.
