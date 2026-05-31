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
    const intervalBaselineWeightMatch = firstMatch(lower, [
      /(?:current\s+approval|approval|authority|funding|funded|interval|4\s*month|four\s*month|last\s+approved|most\s+recent\s+approved)[^\n.]{0,60}(?:baseline|start|starting|previous|initial)?\s*weight\D{0,20}(\d+(?:[.,]\d+)?)\s*kg/,
      /(?:baseline|start|starting|previous)\s+weight\s+(?:for|at|of)\s+(?:current\s+)?(?:approval|authority|funding|funded|interval)\D{0,20}(\d+(?:[.,]\d+)?)\s*kg/,
      /(?:weight\s+at\s+start\s+of\s+(?:current\s+)?(?:approval|authority|funding|interval))\D{0,20}(\d+(?:[.,]\d+)?)\s*kg/
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
    const intervalBaselineWeight = numberValue(intervalBaselineWeightMatch?.[1]);
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
      intervalBaselineWeight,
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

  function todayAu() {
    return new Intl.DateTimeFormat("en-AU", {
      timeZone: "Australia/Perth",
      day: "2-digit",
      month: "2-digit",
      year: "numeric"
    }).format(new Date());
  }

  function extractLine(raw, patterns) {
    const text = String(raw || "");
    for (const pattern of patterns) {
      const match = text.match(pattern);
      if (match?.[1]) return match[1].trim();
    }
    return "";
  }

  function percentWeightLoss(facts) {
    if (!facts.previousWeight || !facts.currentWeight) return null;
    return ((facts.previousWeight - facts.currentWeight) / facts.previousWeight) * 100;
  }

  function vapacIntervalBaselineWeight(facts) {
    return facts.intervalBaselineWeight || null;
  }

  function vapacIntervalWeightLoss(facts) {
    const baseline = vapacIntervalBaselineWeight(facts);
    if (!baseline || !facts.currentWeight) return null;
    return ((baseline - facts.currentWeight) / baseline) * 100;
  }

  function criticalVapacIssues(raw, facts) {
    const lower = String(raw || "").toLowerCase();
    const issues = [];
    if (!extractLine(raw, [/(?:patient|name)\s*[:\-]\s*([^\n]+)/i]) && !/\bmr\.|\bmrs\.|\bms\.|\bmiss\b/i.test(raw)) issues.push("Patient full name/title not clearly documented.");
    if (!/\b\d{1,2}\/\d{1,2}\/\d{2,4}\b/.test(raw) && !/dob|date of birth/i.test(raw)) issues.push("DOB not clearly documented.");
    if (!/gold|white|dva|vsm|file|card/i.test(raw)) issues.push("DVA card type and/or DVA file number not clearly documented.");
    if (!vapacIntervalBaselineWeight(facts)) issues.push("Baseline weight for the most recent 4-month approval/funding interval is not clearly documented. Do not use older original treatment starting weight for the 5% continuation calculation unless it is explicitly the current interval baseline.");
    if (!facts.currentWeight) issues.push("Current weight not clearly documented.");
    if (!facts.heightCm) issues.push("Height not clearly documented, so BMI cannot be verified.");
    if (!/accepted conditions?|comorbid|oa|osa|hypertension|htn|diabetes|dm|pain|back|knee|ankle|mental health|ptsd/i.test(raw)) issues.push("Accepted conditions / relevant comorbidities not clearly documented.");
    if (!facts.medicine && !/tirzepatide|semaglutide|mounjaro|wegovy|ozempic/i.test(lower)) issues.push("Requested medication not clearly documented.");
    if (!facts.plannedDose && !/requested|proposed|maintenance|continue|continuation/i.test(lower)) issues.push("Requested dose/regimen not clearly documented.");
    if (!facts.hasDietician) issues.push("Dietitian/dietician engagement not clearly documented.");
    if (vapacIntervalWeightLoss(facts) !== null && vapacIntervalWeightLoss(facts) < 5) issues.push(`Weight loss across the most recent approval interval is ${formatNumber(vapacIntervalWeightLoss(facts), 1)}%, below the commonly cited 5% continuation threshold; written justification should address clinical benefits/barriers.`);
    return issues;
  }

  function vapacApplication(raw, facts) {
    const issues = criticalVapacIssues(raw, facts);
    const intervalBaseline = vapacIntervalBaselineWeight(facts);
    const intervalLoss = vapacIntervalWeightLoss(facts);
    const lifetimeLoss = percentWeightLoss(facts);
    const patientName = extractLine(raw, [/(?:patient|name)\s*[:\-]\s*([^\n]+)/i])
      || String(raw || "").match(/\b(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+[A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+)+/)?.[0]
      || "Patient name: Not documented";
    const dob = extractLine(raw, [/(?:dob|date of birth)\s*[:\-]\s*([^\n]+)/i])
      || String(raw || "").match(/\b\d{1,2}\/\d{1,2}\/\d{2,4}\b/)?.[0]
      || "DOB: Not documented";
    const dva = extractLine(raw, [/(?:dva|card|file)\s*[:\-]\s*([^\n]+)/i])
      || String(raw || "").match(/\b(?:Gold|White)\b\s*[-:]?\s*[A-Z]{2,}\d+/i)?.[0]
      || "DVA card/file: Not documented";
    const requestedMed = facts.medicine ? `${facts.medicine}${facts.generic ? ` (${facts.generic})` : ""}` : "Not documented";
    const requestedDose = facts.plannedDose || facts.currentDose
      ? `${formatNumber(facts.plannedDose || facts.currentDose, 2)} mg once weekly`
      : "Not documented";

    return [
      "Apex Rx 447 Upper Edward Street",
      "Spring Hill, QLD 4000",
      "Ph: 1300273979",
      "Fax: 0739168300",
      "E: contact@apexrx.com.au",
      "",
      "Department of Veterans' Affairs - Application for Funding of Weight Loss Pharmacotherapy Veterans' Affairs Pharmaceutical Advisory Centre (VAPAC)",
      "",
      todayAu(),
      "",
      "Dear Sirs/ Madams",
      "",
      patientName,
      dob,
      dva,
      "",
      `Starting Weight for Current Approval Interval: ${intervalBaseline ? `${formatNumber(intervalBaseline, 1)} kg` : "Not documented"}`,
      `Current Weight: ${facts.currentWeight ? `${formatNumber(facts.currentWeight, 1)} kg` : "Not documented"}`,
      `Height: ${facts.heightCm ? `${formatNumber(facts.heightCm, 1)} cm` : "Not documented"}`,
      `BMI: ${facts.bmi ? `~${formatNumber(facts.bmi, 1)} kg/m2 (${bmiCategory(facts.bmi)})` : "Not documented"}`,
      "Accepted Conditions / Comorbidities:",
      extractLine(raw, [/(?:accepted conditions?|comorbidities|comorbid conditions)\s*[:\-]?\s*([^\n]+)/i]) || "Not documented",
      "",
      "Clinical Summary (Reason for request):",
      intervalLoss !== null
        ? `Documented weight change across the most recent approval interval is ${formatNumber(intervalLoss, 1)}%. The 5% continuation requirement should be assessed against this interval baseline, not older original treatment weights. ${intervalLoss >= 5 ? "This appears to meet the 5% interval threshold." : "This appears below the 5% interval threshold and requires written clinical justification for continuation."}`
        : "Percentage weight loss for the most recent approval interval cannot be calculated because the interval baseline weight and/or current weight is not clearly supplied.",
      lifetimeLoss !== null && facts.previousWeight !== intervalBaseline ? `Original/older treatment weight change supplied as background: ${formatNumber(lifetimeLoss, 1)}%. This is not used for the 5% interval continuation calculation unless explicitly stated as the current approval interval baseline.` : "",
      raw,
      "",
      "Current Medication (Generic/ Brand name/ Dose):",
      requestedMed === "Not documented" ? "Not documented" : `${requestedMed} ${requestedDose}`,
      "",
      "Medication History:",
      "Product Name    Dosage    Frequency",
      "See supplied medication history below / above. Convert pasted prescribing history into rows in the live AI output.",
      "",
      "Requested Medication:",
      requestedMed,
      "",
      "Proposed dose and regimen:",
      requestedDose,
      "",
      "Intended as a maintenance dose, with ongoing clinical review and dose adjustment if required",
      "Continued alongside lifestyle measures, dietitian input, and physical activity",
      "Planned duration: Ongoing treatment for 4 months, subject to review.",
      "",
      "Monitoring and review:",
      "BMI will be used as the primary objective marker for response, with assessment at regular follow-up intervals. Adjunctive lifestyle measures, including regular exercise and dietitian reviews, will continue. Regular engagement with the doctor for reviews also acts as a form of check-in and behaviour activation, where the doctor can also provide informal psychological and medical support in the form of reassurance.",
      "",
      "Evidence Supporting Efficacy",
      "The following peer-reviewed studies provide evidence supporting the efficacy of once-weekly GLP-1/GIP receptor agonist therapy in adults with overweight or obesity: 'Tirzepatide Once Weekly for the Treatment of Obesity' - New England Journal of Medicine (2022). 'Once-Weekly Semaglutide in Adults with Overweight or Obesity' - New England Journal of Medicine (2021).",
      "",
      "Summary",
      "In participants with overweight or obesity, both tirzepatide and semaglutide, administered once weekly alongside lifestyle interventions, were associated with sustained, clinically significant reductions in body weight and improvements in cardiometabolic markers compared to placebo.",
      "",
      "We request a 4 month prescription of medication.",
      "",
      "Additional Notes",
      "Renal/hepatic status: Not documented unless supplied above.",
      "",
      "Conclusion",
      "This application is made under RPBS arrangements for consideration by the Department of Veterans' Affairs (VAPAC). The requested treatment is clinically appropriate for this patient's presentation and comorbidity profile, with supporting evidence for efficacy and safety.",
      "",
      "Dr Michael Addis",
      "665437AX",
      "contact@apex.au",
      "",
      "Critical information missing / issues to address:",
      ...(issues.length ? issues.map((issue) => `- ${issue}`) : ["- No critical missing information identified from the supplied input."])
    ].join("\n");
  }

  window.buildLocalFallbackNote = function buildLocalFallbackNote(type, raw) {
    const lower = String(type || "").toLowerCase();
    const facts = extractWeightLossFacts(raw);
    if (lower === "vapac weight loss application") {
      return vapacApplication(raw, facts);
    }

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
