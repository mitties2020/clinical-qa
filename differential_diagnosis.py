# differential_diagnosis.py - Intelligent Diagnosis Scoring System
# Ranks diagnoses by probability based on patient presentation
# 18 conditions across 3 complaint types (chest pain, acute abdomen, headache)

import json
from typing import List, Dict

DIAGNOSIS_RULES = {
    'chest_pain': {
        'acute_coronary_syndrome': {
            'base_probability': 0.25,
            'age_weight': {'<40': -0.1, '40-65': 0.3, '>65': 0.2},
            'gender_weight': {'M': 0.1, 'F': -0.05},
            'risk_factors': ['hypertension', 'diabetes', 'smoking', 'hyperlipidemia'],
            'supporting_symptoms': ['diaphoresis', 'dyspnea', 'palpitations', 'chest_pressure'],
            'red_flag': True,
        },
        'pulmonary_embolism': {
            'base_probability': 0.15,
            'age_weight': {'<40': 0.05, '40-65': 0.1, '>65': 0.15},
            'gender_weight': {'M': 0, 'F': 0.1},
            'risk_factors': ['immobility', 'recent_surgery', 'malignancy', 'pregnancy'],
            'supporting_symptoms': ['dyspnea', 'tachycardia', 'hypoxia', 'leg_swelling'],
            'red_flag': True,
        },
        'pneumonia': {
            'base_probability': 0.20,
            'age_weight': {'<40': 0.05, '40-65': 0.15, '>65': 0.25},
            'gender_weight': {'M': 0.05, 'F': 0},
            'risk_factors': ['smoking', 'copd', 'immunosuppression', 'recent_viral'],
            'supporting_symptoms': ['cough', 'fever', 'dyspnea', 'sputum'],
            'red_flag': True,
        },
        'musculoskeletal_pain': {
            'base_probability': 0.25,
            'age_weight': {'<40': 0.1, '40-65': 0.15, '>65': 0.1},
            'gender_weight': {'M': 0.05, 'F': 0.05},
            'risk_factors': ['recent_trauma', 'repetitive_strain', 'poor_posture'],
            'supporting_symptoms': ['reproducible_pain', 'focal_tenderness', 'worse_with_movement'],
            'red_flag': False,
        },
    },
    'acute_abdomen': {
        'appendicitis': {
            'base_probability': 0.15,
            'age_weight': {'<20': 0.2, '20-50': 0.15, '>50': -0.05},
            'risk_factors': [],
            'supporting_symptoms': ['mcburney_point_tenderness', 'fever', 'nausea_vomiting'],
            'red_flag': True,
        },
        'acute_cholecystitis': {
            'base_probability': 0.12,
            'age_weight': {'<40': 0.05, '40-65': 0.15, '>65': 0.12},
            'gender_weight': {'M': -0.05, 'F': 0.15},
            'risk_factors': ['gallstones', 'obesity', 'female', 'family_history'],
            'supporting_symptoms': ['rucherles_sign', 'right_upper_quadrant', 'murphy_sign'],
            'red_flag': True,
        },
        'gastroenteritis': {
            'base_probability': 0.20,
            'age_weight': {'<40': 0.2, '40-65': 0.2, '>65': 0.15},
            'risk_factors': ['recent_travel', 'food_exposure', 'ill_contacts'],
            'supporting_symptoms': ['diarrhea', 'vomiting', 'fever', 'cramping'],
            'red_flag': False,
        },
    },
    'acute_headache': {
        'tension_headache': {
            'base_probability': 0.40,
            'age_weight': {'<40': 0.4, '40-65': 0.45, '>65': 0.35},
            'gender_weight': {'M': 0.3, 'F': 0.5},
            'risk_factors': ['stress', 'poor_posture', 'sleep_deprivation'],
            'supporting_symptoms': ['bilateral', 'pressing', 'neck_tension', 'no_photophobia'],
            'red_flag': False,
        },
        'migraine': {
            'base_probability': 0.25,
            'age_weight': {'<40': 0.3, '40-65': 0.2, '>65': 0.05},
            'gender_weight': {'M': 0.15, 'F': 0.35},
            'risk_factors': ['family_history', 'female', 'hormonal_factors'],
            'supporting_symptoms': ['unilateral', 'pulsating', 'photophobia', 'phonophobia'],
            'red_flag': False,
        },
        'meningitis': {
            'base_probability': 0.05,
            'age_weight': {'<20': 0.15, '20-50': 0.08, '>50': 0.05},
            'risk_factors': ['fever', 'recent_travel', 'immunosuppression'],
            'supporting_symptoms': ['fever', 'neck_stiffness', 'kernigs_sign', 'altered_mental'],
            'red_flag': True,
        },
    },
}

def calculate_diagnosis_score(presenting_complaint, patient_age, patient_gender, comorbidities, symptoms, exam_findings):
    """Calculate probability scores for all diagnoses in differential"""
    
    age_group = '<40' if patient_age < 40 else '40-65' if patient_age < 65 else '>65'
    gender = 'M' if patient_gender.lower().startswith('m') else 'F'
    
    if presenting_complaint.lower() not in DIAGNOSIS_RULES:
        return []
    
    diagnoses = DIAGNOSIS_RULES[presenting_complaint.lower()]
    scored_diagnoses = []
    
    for diagnosis_name, rules in diagnoses.items():
        score = rules['base_probability']
        
        age_adjustment = rules['age_weight'].get(age_group, 0)
        score += age_adjustment
        
        gender_adjustment = rules['gender_weight'].get(gender, 0)
        score += gender_adjustment
        
        matching_risk_factors = [rf for rf in rules['risk_factors'] if rf in comorbidities]
        risk_factor_boost = len(matching_risk_factors) * 0.05
        score += min(risk_factor_boost, 0.25)
        
        matching_symptoms = [s for s in rules['supporting_symptoms'] if s.lower() in [x.lower() for x in symptoms]]
        symptom_boost = len(matching_symptoms) * 0.08
        score += min(symptom_boost, 0.4)
        
        score = max(0, min(score, 1.0))
        
        scored_diagnoses.append({
            'diagnosis': diagnosis_name,
            'probability_score': round(score, 3),
            'probability_percent': round(score * 100),
            'confidence_level': _get_confidence_level(score),
            'red_flag': rules['red_flag'],
            'supporting_symptoms': matching_symptoms,
            'risk_factors_present': matching_risk_factors,
        })
    
    scored_diagnoses.sort(key=lambda x: x['probability_score'], reverse=True)
    return scored_diagnoses

def _get_confidence_level(score):
    if score >= 0.8: return 'VERY HIGH'
    elif score >= 0.6: return 'HIGH'
    elif score >= 0.4: return 'MODERATE'
    elif score >= 0.2: return 'LOW'
    else: return 'VERY LOW'
