# Australian Clinical Guidelines Database
# TGA, NHMRC, Therapeutic Guidelines references

GUIDELINES_DB = {
    "respiratory": {
        "pneumonia": {
            "title": "Community-Acquired Pneumonia (CAP)",
            "sources": [
                {
                    "name": "Therapeutic Guidelines: Respiratory",
                    "url": "https://www.tg.org.au/",
                    "key_points": [
                        "CURB-65 severity assessment required",
                        "Empiric amoxicillin or doxycycline for mild CAP",
                        "Amoxicillin-clavulanate or respiratory fluoroquinolone for moderate",
                        "Consider atypical coverage if risk factors present"
                    ]
                },
                {
                    "name": "NHMRC Guideline on Lower Respiratory Tract Infection",
                    "url": "https://www.nhmrc.gov.au/",
                    "key_points": [
                        "Chest X-ray recommended for suspected CAP",
                        "Blood cultures if hospitalized",
                        "Vaccination: pneumococcal + influenza"
                    ]
                }
            ]
        },
        "asthma": {
            "title": "Asthma Management",
            "sources": [
                {
                    "name": "Therapeutic Guidelines: Respiratory (Asthma)",
                    "url": "https://www.tg.org.au/",
                    "key_points": [
                        "ICS-formoterol as reliever (MART regimen)",
                        "ICS + LABA maintenance for persistent asthma",
                        "Avoid SABA monotherapy in mild persistent asthma",
                        "Annual asthma action plan review"
                    ]
                }
            ]
        }
    },
    "infection": {
        "uti": {
            "title": "Urinary Tract Infection (UTI)",
            "sources": [
                {
                    "name": "Therapeutic Guidelines: Antibiotic",
                    "url": "https://www.tg.org.au/",
                    "key_points": [
                        "Uncomplicated cystitis: Nitrofurantoin 100mg BD x 5 days OR Trimethoprim 300mg BD x 5 days",
                        "Avoid fluoroquinolones for uncomplicated UTI (first-line reserve)",
                        "Pregnancy: Cephalexin or amoxicillin preferred (avoid nitrofurantoin at term)",
                        "Pyelonephritis: Requires systemic therapy, consider hospitalization if severe"
                    ]
                },
                {
                    "name": "NHMRC: Asymptomatic Bacteriuria",
                    "url": "https://www.nhmrc.gov.au/",
                    "key_points": [
                        "Do NOT treat asymptomatic bacteriuria in non-pregnant patients",
                        "Treat in pregnancy only"
                    ]
                }
            ]
        },
        "tb": {
            "title": "Tuberculosis Management",
            "sources": [
                {
                    "name": "NHMRC: Tuberculosis Management in Australia",
                    "url": "https://www.nhmrc.gov.au/",
                    "key_points": [
                        "4-drug regimen: HRZE (isoniazid, rifampicin, pyrazinamide, ethambutol) x 2 months",
                        "Follow-up: HR x 4 months",
                        "Public health notification mandatory",
                        "Contact tracing required",
                        "Directly-observed therapy (DOT) recommended"
                    ]
                }
            ]
        }
    },
    "cardiovascular": {
        "hypertension": {
            "title": "Hypertension Management",
            "sources": [
                {
                    "name": "National Heart Foundation of Australia: Guideline on Hypertension",
                    "url": "https://heartfoundation.org.au/",
                    "key_points": [
                        "Target BP <140/90 mmHg for most adults",
                        "Target <130/80 mmHg if diabetes or CKD",
                        "First-line: ACE-I/ARB, CCB, or thiazide diuretic",
                        "Avoid NSAIDs in hypertension"
                    ]
                }
            ]
        }
    },
    "gastro": {
        "gord": {
            "title": "Gastro-Oesophageal Reflux Disease (GORD)",
            "sources": [
                {
                    "name": "Therapeutic Guidelines: Gastrointestinal",
                    "url": "https://www.tg.org.au/",
                    "key_points": [
                        "Lifestyle modification: Avoid triggers, elevation, weight loss",
                        "Trial PPI x 4 weeks: Omeprazole 20mg daily or Pantoprazole 40mg daily",
                        "H2-receptor antagonist (ranitidine removed from TGA approval)",
                        "Long-term PPI use: Monitor for B12, magnesium, fracture risk"
                    ]
                }
            ]
        }
    },
    "endocrine": {
        "diabetes_t2": {
            "title": "Type 2 Diabetes Management",
            "sources": [
                {
                    "name": "Therapeutic Guidelines: Endocrinology",
                    "url": "https://www.tg.org.au/",
                    "key_points": [
                        "HbA1c target: 7% (53 mmol/mol) for most; individualize for elderly/high risk",
                        "Metformin first-line if tolerated",
                        "GLP-1 RA if cardiovascular disease or weight loss needed",
                        "SGLT2i: For CKD or heart failure",
                        "Annual: Urine albumin, renal function, lipids, eye screening"
                    ]
                },
                {
                    "name": "RACGP: General Practice Management of Type 2 Diabetes",
                    "url": "https://www.racgp.org.au/",
                    "key_points": [
                        "Structured diabetes education mandatory",
                        "Annual diabetic foot assessment",
                        "Influenza vaccination annually"
                    ]
                }
            ]
        }
    }
}

def get_guidelines_for_condition(condition_keyword: str) -> list:
    """Search guidelines DB for condition"""
    results = []
    for category, conditions in GUIDELINES_DB.items():
        for cond_key, cond_data in conditions.items():
            if condition_keyword.lower() in cond_key.lower() or condition_keyword.lower() in cond_data.get("title", "").lower():
                results.append({
                    "category": category,
                    "condition": cond_key,
                    "data": cond_data
                })
    return results

def format_guidelines_for_response(guidelines: list) -> str:
    """Format guidelines for inclusion in AI response"""
    if not guidelines:
        return ""
    
    output = "\n\n---\n**AUSTRALIAN CLINICAL GUIDELINES REFERENCE:**\n"
    
    for guideline in guidelines:
        cond_data = guideline["data"]
        output += f"\n**{cond_data['title']}**\n"
        
        for source in cond_data.get("sources", []):
            output += f"\n*Source: {source['name']}*\n"
            if source.get("url"):
                output += f"[View Guideline]({source['url']})\n"
            
            output += "Key recommendations:\n"
            for point in source.get("key_points", []):
                output += f"• {point}\n"
    
    return output


# Template library for common documents
DOCUMENT_TEMPLATES = {
    "referral_letter": {
        "name": "Medical Referral Letter",
        "template": """REFERRAL TO [SPECIALTY]

Date: [DATE]
From: Dr [YOUR NAME]
To: [SPECIALIST NAME]
Re: [PATIENT NAME] (DOB: [DOB])

CLINICAL SUMMARY:
[BRIEF PRESENTATION AND HISTORY]

RELEVANT INVESTIGATIONS:
[TEST RESULTS]

CURRENT MEDICATIONS:
[LIST MEDICATIONS]

REASON FOR REFERRAL:
[WHY REFERRED, SPECIFIC QUESTIONS FOR SPECIALIST]

URGENT/ROUTINE:
[TICK AS APPROPRIATE]

---
Dr [NAME]
[QUALIFICATIONS]
[CONTACT DETAILS]
"""
    },
    "discharge_summary": {
        "name": "Hospital Discharge Summary",
        "template": """DISCHARGE SUMMARY

PATIENT: [NAME] | DOB: [DOB] | MRN: [MRN]
ADMISSION DATE: [DATE]
DISCHARGE DATE: [DATE]
CONSULTANT: Dr [NAME]

DIAGNOSIS:
1. [PRIMARY DIAGNOSIS]
2. [SECONDARY DIAGNOSES]

PROCEDURE/TREATMENT:
[WHAT WAS DONE]

FINDINGS:
[KEY CLINICAL/INVESTIGATION FINDINGS]

MEDICATIONS ON DISCHARGE:
[CURRENT MEDICATIONS WITH DOSES]

FOLLOW-UP:
- GP review in [TIMEFRAME]
- Specialist follow-up: [IF NEEDED]
- Investigations pending: [IF ANY]

SAFETY ALERTS:
[ALLERGIES, DRUG INTERACTIONS, PRECAUTIONS]

---
Dr [NAME]
"""
    },
    "medical_certificate": {
        "name": "Medical Certificate (Work/School)",
        "template": """MEDICAL CERTIFICATE

PATIENT NAME: [NAME]
DATE OF BIRTH: [DOB]
DATE OF CONSULTATION: [DATE]

I certify that I have examined [PATIENT NAME] on [DATE] and in my opinion:

DIAGNOSIS: [CONDITION - can be withheld]

This patient is unfit for work/school from [START DATE] to [END DATE] (inclusive).

Days absent: [NUMBER OF DAYS]

NOTES:
[ANY SPECIFIC RESTRICTIONS OR RECOMMENDATIONS]

Fitness to return: [DATE OR CONDITIONAL]

This certificate is issued in accordance with the National Health Act 1953.

---
Dr [NAME]
[QUALIFICATIONS]
[REGISTRATION NUMBER]
[PRACTICE ADDRESS]
[CONTACT NUMBER]
"""
    },
    "clinical_note": {
        "name": "Structured Clinical Note (SOAP Format)",
        "template": """CLINICAL NOTE

PATIENT: [NAME] | DOB: [DOB] | DATE: [DATE]

SUBJECTIVE:
[PATIENT'S PRESENTING COMPLAINT AND HISTORY]

OBJECTIVE:
Vitals: BP [__], HR [__], RR [__], Temp [__], SpO2 [__]
Examination findings: [PHYSICAL EXAM FINDINGS]
Investigations: [RELEVANT TEST RESULTS]

ASSESSMENT:
[CLINICAL IMPRESSION AND DIAGNOSIS]

PLAN:
1. Investigation: [WHAT TESTS/IMAGING ORDERED]
2. Treatment: [MEDICATIONS, PROCEDURES]
3. Referral: [IF NEEDED]
4. Follow-up: [WHEN AND WITH WHOM]
5. Patient education: [WHAT DISCUSSED]

---
Dr [NAME]
"""
    },
    "incident_report": {
        "name": "Clinical Incident Report",
        "template": """INCIDENT REPORT

INCIDENT DATE & TIME: [DATE/TIME]
LOCATION: [WHERE IT OCCURRED]
REPORTER: Dr [NAME] | DATE REPORTED: [DATE]

PATIENT INVOLVED:
Name: [NAME] | DOB: [DOB] | MRN: [MRN]

DESCRIPTION OF INCIDENT:
[FACTUAL ACCOUNT OF WHAT HAPPENED]

CONTRIBUTING FACTORS:
[SYSTEMIC OR SITUATIONAL FACTORS]

HARM ASSESSMENT:
No harm / Minor harm / Moderate harm / Severe harm

IMMEDIATE ACTIONS TAKEN:
[WHAT WAS DONE TO MANAGE SITUATION]

CORRECTIVE ACTIONS:
[PREVENTIVE MEASURES TO AVOID RECURRENCE]

---
Reporter: Dr [NAME]
Clinical Lead Review: Dr [NAME] | Date: [DATE]
"""
    }
}

def get_template(template_name: str) -> dict:
    """Get document template"""
    return DOCUMENT_TEMPLATES.get(template_name, {})

def list_templates() -> list:
    """List all available templates"""
    return [{"key": k, "name": v["name"]} for k, v in DOCUMENT_TEMPLATES.items()]
