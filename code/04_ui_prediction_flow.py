# Excerpt from the private Streamlit app (app_public.py).
# The app validates both uploads, then passes them to the shared inference
# service. The optional LIME explanation is requested with a learner ID, and
# its quality warning is shown next to it. Not runnable on its own: the
# upload widgets, `load_uploads`, `service`, and `MAX_LEARNERS` come from the
# private app.

import streamlit as st

# Validate both CSVs and enforce the two-learner limit.
learners, activity = load_uploads(
    learners_file.getvalue(), activity_file.getvalue(), max_learners=MAX_LEARNERS
)

# Run the model and, if an ID was entered, a LIME explanation for that learner.
result = service().predict(learners, activity, explain_id or None)

# A quality warning is shown next to any LIME explanation.
explanation = result["lime_explanation"]
if explanation["quality_warning"]:
    st.warning(explanation["quality_warning"])
