# medical_guidelines.py - Australian Medical Guidelines Reference
# DVA entitlements, contraindications, safety netting

DVA_CONDITIONS = {
    'musculoskeletal': {
        'lower_back_pain': {'physio': True, 'psychology': False},
        'osteoarthritis': {'physio': True, 'psychology': False},
        'neck_pain': {'physio': True, 'psychology': False},
    },
    'mental_health': {
        'ptsd': {'physio': False, 'psychology': True},
        'depression': {'physio': False, 'psychology': True},
        'anxiety': {'physio': False, 'psychology': True},
    },
}

CONTRAINDICATIONS = {
    'paracetamol': ['severe_liver_disease', 'chronic_alcohol_use'],
    'nsaids': ['peptic_ulcer', 'renal_impairment', 'heart_failure', 'pregnancy_3rd'],
    'ace_inhibitors': ['hyperkalemia', 'bilateral_renal_artery_stenosis', 'pregnancy'],
    'metformin': ['severe_renal_impairment', 'acute_illness', 'lactic_acidosis'],
}

SAFETY_NETTING = {
    'worsening_pain': 'Escalate if pain persists >72hrs',
    'fever': 'Review if fever develops',
    'neurological_signs': 'URGENT: limb weakness, speech changes → emergency',
    'chest_pain': 'URGENT: chest pain → rule out cardiac causes',
}

def check_contraindications(medication, conditions):
    """Check for drug-condition contraindications"""
    med_lower = medication.lower()
    found = []
    for med, contraindicated_conditions in CONTRAINDICATIONS.items():
        if med in med_lower:
            for condition in contraindicated_conditions:
                if condition in [c.lower() for c in conditions]:
                    found.append(f"CONTRAINDICATION: {med} may worsen {condition}")
    return found

def get_dva_entitlements(condition):
    """Get DVA entitlements for a condition"""
    condition_lower = condition.lower()
    for category, conditions in DVA_CONDITIONS.items():
        for cond_key in conditions.keys():
            if cond_key in condition_lower:
                return {'category': category, 'condition': cond_key, 'entitlements': conditions[cond_key]}
    return None
