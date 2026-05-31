(function () {
  function numberValue(value) {
    const parsed = Number(String(value || "").replace(",", "."));
    return Number.isFinite(parsed) ? parsed : null;
  }

  function firstMatch(text, patterns) {
    for (const pattern of patterns) {
      const match = text.match(pattern);
      if (match) return match;
    }
    return null;
  }

  function formatNumber(value, decimals) {
    if (value === null || value === undefined || Number.isNaN(value)) return "";
    return Number(value).toFixed(decimals).replace(/\.0$/, "");
  }

  function bmiCategory(bmi) {
    if (!bmi) return "";
    if (bmi >= 40) return "Class III obesity";
    if (bmi >= 35) return "Class II obesity";
    if (bmi >= 30) return "Class I obesity";
    if (bmi >= 25) return "overweight range";
    return "healthy weight range";
  }

  function extractWeightLossFacts(raw) {
    const text = String(raw || "");
    const lower = text.toLowerCase();
    const ageMatch = firstMatch(lower, [
      /(\d{1,3})\s*(?:year old|years old|yr old|yo|y\/o)\s*(man|male|woman|female)?/,
      /(\d{1,3})\s*(?:m|f)\b/
    ]);
    const currentWeightMatch = firstMatch(lower, [
      /(?:today'?s|todays|current|now)\s+weight\D{0,20}(\d+(?:[.,]\d+)?)\s*kg/,
      /(?:weight)\D{0,12}(\d+(?:[.,]\d+)?)\s*kg/
    ]);
    const previousWeightMatch = firstMatch(lower, [
      /(?:last|previous|starting|initial)\s+weight\D{0,20}(\d+(?:[.,]\d+)?)\s*kg/
    ]);
    const heightMatch = firstMatch(lower, [
      /(?:height|heaight|ht)\D{0,12}(\d+(?:[.,]\d+)?)\s*cm/
    ]);
    const currentDoseMatch = firstMatch(lower, [
      /(?:ozempic|semaglutide|semagulide|mounjaro|tirzepatide)[^\d]{0,20}(\d+(?:[.,]\d+)?)\s*mg/,
      /(?:current|on)\s+dose\D{0,20}(\d+(?:[.,]\d+)?)\s*mg/
    ]);
    const plannedDoseMatch = firstMatch(lower, [
      /increase\s+(?:ozempic|semaglutide|semagulide|mounjaro|tirzepatide)?\s*(?:to\s+)?(\d+(?:[.,]\d+)?)\s*mg/,
      /(?:plan|commence|start|increase)[^\d]{0,40}(\d+(?:[.,]\d+)?)\s*mg/
    ]);
    const reviewMatch = lower.match(/review\s+in\s+(\d+)\s*weeks?/);
    const durationMatch = lower.match(/for\s+(\d+)\s*weeks?/);

    const currentWeight = numberValue(currentWeightMatch?.[1]);
    const previousWeight = numberValue(previousWeightMatch?.[1]);
    const heightCm = numberValue(heightMatch?.[1]);
    const currentDose = numberValue(currentDoseMatch?.[1]);
    const plannedDose = numberValue(plannedDoseMatch?.[1]);
    const bmi = currentWeight && heightCm ? currentWeight / ((heightCm / 100) ** 2) : null;

    let medicine = "";
    let generic = "";
    if (/mounjaro|tirzepatide/.test(lower)) {
      medicine = "Mounjaro";
      generic = "tirzepatide";
    } else if (/ozempic|semaglutide|semagulide/.test(lower)) {
      medicine = "Ozempic";
      generic = "semaglutide";
    }

    return {
      age: ageMatch?.[1] || "",
      sex: ageMatch?.[2] || "",
      currentWeight,
      previousWeight,
      heightCm,
      bmi,
      medicine,
      generic,
      currentDose,
      plannedDose,
      reviewWeeks: reviewMatch?.[1] || "",
      durationWeeks: durationMatch?.[1] || "",
      hasDietician: /dietician|dietitian/.test(lower),
      hasExercise: /exercis/.test(lower),
      appetiteControlled: /appetite[^.\n]*(well|wel)?\s*controlled|appetite suppression|appetite[^.\n]*suppressed/.test(lower),
      noSideEffects: /no\s+(significant\s+)?side effects?|nil\s+(significant\s+)?side effects?/.test(lower),
      safetyNettingProvided: /safety\s*net/.test(lower),
      obesity: /obesity|obese/.test(lower)
    };
  }

  function patientProfile(facts) {
    const ageSex = [facts.age ? `${facts.age} year old` : "", facts.sex || ""].filter(Boolean).join(" ");
    if (ageSex && facts.obesity) return `${ageSex} with obesity.`;
    if (ageSex) return `${ageSex}.`;
    return facts.obesity ? "Patient with obesity." : "Not documented";
  }

  function anthropometrics(facts) {
    const lines = [];
    if (facts.previousWeight) lines.push(`Previous weight: ${formatNumber(facts.previousWeight, 1)} kg`);
    lines.push(`Current weight: ${facts.currentWeight ? `${formatNumber(facts.currentWeight, 1)} kg` : "Not documented"}`);
    lines.push(`Height: ${facts.heightCm ? `${formatNumber(facts.heightCm, 1)} cm` : "Not documented"}`);
    lines.push(`BMI: ${facts.bmi ? `~${formatNumber(facts.bmi, 1)} kg/m2 (${bmiCategory(facts.bmi)})` : "Not documented"}`);
    return lines;
  }

  function currentTreatment(facts) {
    if (!facts.medicine && !facts.currentDose) return ["Medication and current dose: Not documented"];
    return [
      `Medication: ${facts.medicine ? `${facts.medicine}${facts.generic ? ` (${facts.generic})` : ""}` : "Not documented"}`,
      `Current dose: ${facts.currentDose ? `${formatNumber(facts.currentDose, 2)} mg once weekly` : "Not documented"}`
    ];
  }

  function progressLines(facts, raw) {
    const lines = [];
    if (facts.previousWeight && facts.currentWeight) {
      lines.push(`Weight reduced from ${formatNumber(facts.previousWeight, 1)} kg to ${formatNumber(facts.currentWeight, 1)} kg.`);
    }
    if (facts.appetiteControlled) lines.push("Appetite well controlled / appetite suppression reported.");
    if (facts.hasDietician) lines.push("Continuing dietician input.");
    if (facts.hasExercise) lines.push("Exercising regularly.");
    if (!lines.length) lines.push(raw);
    return lines;
  }

  function sideEffectLines(facts) {
    return facts.noSideEffects
      ? ["No significant side effects reported."]
      : ["Side effects: Not documented"];
  }

  function planLines(facts) {
    const lines = [];
    if (facts.plannedDose) {
      const med = facts.generic || facts.medicine || "weight-management medication";
      lines.push(`Increase ${med} to ${formatNumber(facts.plannedDose, 2)} mg once weekly${facts.durationWeeks ? ` for ${facts.durationWeeks} weeks` : ""}.`);
      lines.push(`Issue script for ${facts.medicine || med} ${formatNumber(facts.plannedDose, 2)} mg.`);
    } else {
      lines.push("Dose/script plan: Not documented");
    }
    if (facts.hasDietician) lines.push("Continue dietician input.");
    lines.push("Reinforce high-protein diet, hydration, and ongoing lifestyle/activity measures.");
    lines.push(facts.reviewWeeks ? `Review in ${facts.reviewWeeks} weeks.` : "Follow-up timing: Not documented");
    return lines;
  }

  function currentMedicationLine(facts) {
    const dose = facts.plannedDose || facts.currentDose;
    if (!facts.generic && !facts.medicine && !dose) return "Exact prescribing line: Not documented";
    const drug = facts.generic === "tirzepatide" ? "Tirzepatide" : facts.generic === "semaglutide" ? "Semaglutide" : (facts.medicine || "Weight-management medication");
    return `${drug} Subcutaneous Solution Pen-injector ${dose ? `${formatNumber(dose, 2)} mg` : "dose not documented"}, inject weekly, Once a week, , As directed`;
  }

  window.buildLocalFallbackNote = function buildLocalFallbackNote(type, raw) {
    const lower = String(type || "").toLowerCase();
    const facts = extractWeightLossFacts(raw);
    if (lower === "weight loss follow-up") {
      return [
        `Consult Type: Script Renewal - Weight Management${facts.medicine ? ` (${facts.medicine})` : ""}`,
        "Clinician: Not documented",
        "Mode: Telehealth",
        "",
        "ID Verification",
        "Not documented",
        "",
        "Reason for Consult",
        "Review of response to weight-management therapy and script renewal / dose adjustment.",
        "",
        "Medical conditions",
        facts.obesity ? "Obesity" : "Not documented",
        "",
        "Anthropometrics",
        ...anthropometrics(facts),
        "",
        "Current Treatment",
        ...currentTreatment(facts),
        "",
        "Progress Since Last Review",
        ...progressLines(facts, raw),
        "",
        "Side Effects",
        ...sideEffectLines(facts),
        "",
        "Assessment",
        facts.bmi ? `BMI remains in the ${bmiCategory(facts.bmi)} range and ongoing weight-management treatment is clinically indicated.` : "Ongoing weight-management treatment reviewed.",
        facts.currentWeight && facts.previousWeight ? "Positive response with interval weight reduction." : "Treatment response requires confirmation from documented weight trend.",
        facts.noSideEffects ? "No significant adverse effects documented." : "Tolerability requires confirmation.",
        facts.plannedDose ? "Appropriate to proceed with documented dose escalation, subject to clinician confirmation." : "Dose escalation / renewal plan not fully documented.",
        "",
        "Plan",
        ...planLines(facts),
        "",
        "Safety Netting",
        facts.safetyNettingProvided ? "Safety-netting provided." : "Safety-netting not explicitly documented.",
        "Patient advised to stop injections and seek urgent medical care if experiencing persistent vomiting, severe abdominal pain especially right upper quadrant, or symptoms suggestive of pancreatitis/gallbladder pathology.",
        "",
        "Current Medication",
        currentMedicationLine(facts)
      ].join("\n");
    }

    if (lower === "weight loss initial consult") {
      return [
        "Telehealth consult",
        "",
        "ID Verification",
        "Not documented",
        "",
        "Patient profile",
        patientProfile(facts),
        "",
        "DVA / entitlement context",
        "Not documented",
        "",
        "Past medical history",
        facts.obesity ? "Obesity" : "Not documented",
        "",
        "Medications",
        facts.medicine || facts.currentDose ? currentTreatment(facts).join("\n") : "Not documented",
        "",
        "Weight management history",
        raw,
        "",
        "Counselling",
        "Counselled on GLP-1/GIP weight-management therapy where clinically appropriate: once-weekly self-administered subcutaneous injection, appetite/satiety effect, metabolic effect, potential GI side effects, pen/needle administration, and refrigeration/storage requirements.",
        "",
        "Opportunity to ask questions",
        "Not documented",
        "",
        "Assessment",
        facts.bmi ? `BMI ~${formatNumber(facts.bmi, 1)} kg/m2 (${bmiCategory(facts.bmi)}). Weight-management treatment discussed.` : "Initial weight-management consultation completed.",
        "",
        "Plan",
        ...planLines(facts),
        "",
        "Safety Netting",
        facts.safetyNettingProvided ? "Safety-netting provided." : "Safety-netting not explicitly documented.",
        "Advise patient to stop injections and seek urgent care for persistent vomiting, severe abdominal pain especially right upper quadrant, or symptoms suggestive of pancreatitis/gallbladder pathology.",
        "",
        "Observation",
        ...anthropometrics(facts),
        "",
        "Current Medication",
        currentMedicationLine(facts)
      ].join("\n");
    }

    return `Consult type: ${type}\n\nClinical note:\n${raw}\n\nAssessment:\n[Draft assessment generated from input.]\n\nPlan:\n[Draft plan generated from input.]`;
  };
})();
