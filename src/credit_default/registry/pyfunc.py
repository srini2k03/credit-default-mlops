"""MLflow pyfunc wrapper so the registered model also works with generic MLflow tooling."""

from __future__ import annotations

import mlflow.pyfunc


class CreditDefaultPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from credit_default.models.bundle import ModelBundle

        self.bundle = ModelBundle.load(context.artifacts["bundle"])

    def predict(self, context, model_input, params=None):
        import pandas as pd

        proba = self.bundle.predict_proba(model_input)
        return pd.DataFrame(
            {
                "probability": proba,
                "default_prediction": (proba >= self.bundle.threshold).astype(int),
            }
        )
