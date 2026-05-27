const form = document.getElementById("prediction-form");
const resultState = document.getElementById("result-state");
const resultData = document.getElementById("result-data");

const fields = {
  probability: document.getElementById("probability"),
  alert: document.getElementById("alert"),
  threshold: document.getElementById("threshold"),
  riskLevel: document.getElementById("risk-level"),
  modelVersion: document.getElementById("model-version"),
};

const numericFields = new Set([
  "patient_age",
  "weight_kg",
  "height_cm",
  "bmi",
  "icu_los_days_at_obs",
  "charlson_index",
  "hr_mean",
  "hr_min",
  "hr_max",
  "hr_std",
  "sbp_mean",
  "sbp_min",
  "sbp_max",
  "map_mean",
  "map_min",
  "rr_mean",
  "rr_max",
  "temp_mean",
  "temp_max",
  "spo2_mean",
  "spo2_min",
  "gcs_score",
  "sofa_score",
  "news2_score",
  "lactate",
  "wbc",
  "creatinine",
  "bilirubin",
  "platelet_count",
  "sodium",
  "potassium",
  "hemoglobin",
  "pao2_fio2_ratio",
  "fluid_balance_24h",
  "has_diabetes",
  "has_hypertension",
  "has_heart_failure",
  "has_copd",
  "has_ckd",
  "has_immunosuppression",
  "on_vasopressor",
  "on_mechanical_vent",
  "on_rrt",
]);

function buildPayload(formData) {
  const payload = {};

  for (const [key, value] of formData.entries()) {
    if (numericFields.has(key)) {
      payload[key] = value === "" ? null : Number(value);
    } else {
      payload[key] = value;
    }
  }

  return payload;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  resultState.textContent = "Scoring observation...";
  resultData.classList.add("hidden");

  const payload = buildPayload(new FormData(form));

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error("Prediction request failed");
    }

    const data = await response.json();
    fields.probability.textContent = `${(data.deterioration_probability * 100).toFixed(1)}%`;
    fields.alert.textContent = data.alert_triggered ? "Triggered" : "Not Triggered";
    fields.threshold.textContent = data.alert_threshold.toFixed(2);
    fields.riskLevel.textContent = data.risk_level;
    fields.modelVersion.textContent = data.model_version;

    resultState.textContent = "Prediction completed.";
    resultData.classList.remove("hidden");
  } catch (error) {
    resultState.textContent = "The prediction could not be completed. Check that the trained model artifacts exist.";
  }
});

