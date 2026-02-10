"""
PHI Data Filling Tool with JSON Output

This script fills PHI placeholders (___) in medical texts and outputs JSON format
with detailed tracking of all PHI replacements:
  - original_text: the original text with ___ placeholders
  - filled_text: the text with PHI filled in
  - pii_mappings: mapping of each unique filled value to its PHI type
  - filled_values: ordered list of all filled values (preserves duplicates)

Special handling:
  - Follow-up instructions: only extracts actual PHI (doctor names, dates) from
    the generated text, NOT the entire follow-up sentence.
  - Remaining ___: uses context detection (date keywords vs name keywords) to
    determine whether to fill with a date or a name.
  - Dr. ___: fills with just a last name (e.g., "Dr. Chen"), each occurrence
    gets a unique name.
"""

import csv
import json
import re
from datetime import datetime, timedelta
from faker import Faker
import random
import ollama
from typing import Optional, Dict, List, Tuple
from collections import OrderedDict

# Initialize Faker generator
fake = Faker()
Faker.seed(42)
random.seed(42)

# LLM Configuration
MODEL_NAME = "llama3.1"  # Change this to your preferred model
USE_LLM = True  # Set to False to fall back to Faker only

# Probability settings for LLM vs Faker
LLM_PROBABILITY_PATIENT_NAME = 0.5
LLM_PROBABILITY_DOCTOR_NAME = 0.5
LLM_PROBABILITY_HOSPITAL = 0.5
LLM_PROBABILITY_FACILITY = 0.5
LLM_PROBABILITY_FOLLOWUP = 1.0

# Context keywords for determining remaining ___ type
DATE_CONTEXT_KEYWORDS = {
    'on', 'from', 'dated', 'since', 'until', 'before', 'after',
    'date', 'born', 'performed', 'obtained', 'completed', 'done'
}
NAME_CONTEXT_KEYWORDS = {
    'to', 'by', 'with', 'see', 'called', 'contact', 'reported',
    'between', 'notify', 'informed', 'discussed'
}


class LLMPHIGenerator:
    """Use LLM to generate more realistic PHI values"""

    @staticmethod
    def generate_hospital_name(context: str = "") -> str:
        """Use LLM to generate hospital name"""
        if not USE_LLM:
            return f"{fake.city()} {random.choice(['General Hospital', 'Medical Center', 'Regional Hospital', 'Community Hospital'])}"

        prompt = f"""Generate a realistic synthetic hospital name. It should sound like a real hospital but be completely fictional.

Examples of good hospital names:
- "Riverside Medical Center"
- "Memorial General Hospital" 
- "University of [City] Medical Center"
- "St. [Name] Hospital"
- "[City] Regional Health Center"

Generate ONE hospital name only (no quotes, no explanation, just the name):"""

        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.8, "num_predict": 50}
            )
            hospital_name = response["message"]["content"].strip()
            hospital_name = re.sub(r'^["\']|["\']$', '', hospital_name)
            hospital_name = hospital_name.split('\n')[0].strip()
            if not hospital_name:
                raise ValueError("Empty response")
            return hospital_name
        except Exception as e:
            print(f"Warning: LLM failed for hospital name: {e}, using Faker fallback")
            return f"{fake.city()} {random.choice(['General Hospital', 'Medical Center', 'Regional Hospital'])}"

    @staticmethod
    def generate_facility_name(context: str = "") -> str:
        """Use LLM to generate healthcare facility name"""
        if not USE_LLM:
            facilities = [
                "Home Health Services", "Visiting Nurse Association",
                "Community Health Services", "Home Care Partners"
            ]
            return random.choice(facilities)

        prompt = f"""Generate a realistic synthetic home health care facility name. It should sound like a real facility but be completely fictional.

Examples of good facility names:
- "Riverside Home Health Services"
- "Community Care Partners"
- "Premier Home Health Care"
- "Compassionate Care Services"

Generate ONE facility name only (no quotes, no explanation, just the name):"""

        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.8, "num_predict": 50}
            )
            facility_name = response["message"]["content"].strip()
            facility_name = re.sub(r'^["\']|["\']$', '', facility_name)
            facility_name = facility_name.split('\n')[0].strip()
            if not facility_name:
                raise ValueError("Empty response")
            return facility_name
        except Exception as e:
            print(f"Warning: LLM failed for facility name: {e}, using Faker fallback")
            facilities = ["Home Health Services", "Visiting Nurse Association", "Community Health Services"]
            return random.choice(facilities)

    @staticmethod
    def generate_patient_name_llm(sex: Optional[str] = None) -> str:
        """Use LLM to generate patient name"""
        sex_hint = ""
        if sex == 'F':
            sex_hint = " (female name)"
        elif sex == 'M':
            sex_hint = " (male name)"

        prompt = f"""Generate a realistic synthetic patient name. Format: "[First Name] [Last Name]"{sex_hint}

The name should sound realistic but be completely fictional.
Generate ONE patient name in the format "FirstName LastName" (no quotes, no explanation):"""

        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.7, "num_predict": 50}
            )
            name = response["message"]["content"].strip()
            name = re.sub(r'^["\']|["\']$', '', name)
            name = name.split('\n')[0].strip()
            if not name or len(name.split()) < 2:
                raise ValueError("Invalid format")
            return name
        except Exception as e:
            print(f"Warning: LLM failed for patient name: {e}, using Faker fallback")
            if sex == 'F':
                return f"{fake.first_name_female()} {fake.last_name()}"
            elif sex == 'M':
                return f"{fake.first_name_male()} {fake.last_name()}"
            else:
                return f"{fake.first_name()} {fake.last_name()}"

    @staticmethod
    def generate_doctor_name(sex: Optional[str] = None, context: str = "") -> str:
        """Use LLM to generate doctor name (more realistic combinations)"""
        if not USE_LLM:
            if sex == 'F':
                return f"Dr. {fake.first_name_female()} {fake.last_name()}"
            elif sex == 'M':
                return f"Dr. {fake.first_name_male()} {fake.last_name()}"
            else:
                return f"Dr. {fake.first_name()} {fake.last_name()}"

        prompt = f"""Generate a realistic synthetic doctor name. Format: "Dr. [First Name] [Last Name]"

The name should sound professional and realistic but be completely fictional.
Generate ONE doctor name in the exact format "Dr. FirstName LastName" (no quotes, no explanation):"""

        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.7, "num_predict": 50}
            )
            doctor_name = response["message"]["content"].strip()
            doctor_name = re.sub(r'^["\']|["\']$', '', doctor_name)
            doctor_name = doctor_name.split('\n')[0].strip()
            if not doctor_name.startswith("Dr. "):
                doctor_name = "Dr. " + doctor_name
            if not doctor_name or len(doctor_name.split()) < 3:
                raise ValueError("Invalid format")
            return doctor_name
        except Exception as e:
            print(f"Warning: LLM failed for doctor name: {e}, using Faker fallback")
            if sex == 'F':
                return f"Dr. {fake.first_name_female()} {fake.last_name()}"
            elif sex == 'M':
                return f"Dr. {fake.first_name_male()} {fake.last_name()}"
            else:
                return f"Dr. {fake.first_name()} {fake.last_name()}"

    @staticmethod
    def generate_followup_instructions(attending_doctor: str, patient_context: str = "") -> str:
        """Use LLM to generate more realistic follow-up instructions"""
        if not USE_LLM:
            return f"Follow up with {attending_doctor} in 2 weeks"

        doctor_name = attending_doctor.replace("Dr. ", "").strip()

        prompt = f"""Generate realistic follow-up instructions for a patient. The instructions should be natural, professional, and specific.

IMPORTANT: You MUST use the exact attending physician name provided below. Do NOT generate a different doctor name.

Attending physician: {attending_doctor}
Patient context: {patient_context[:200] if patient_context else "General medical follow-up"}

Generate realistic follow-up instructions that MUST mention "{attending_doctor}" (use this exact name). 

Examples of good follow-up instructions:
- "Follow up with {attending_doctor} in 2 weeks for routine check-up"
- "Schedule a follow-up appointment with {attending_doctor} in 1-2 weeks to monitor progress"
- "Please follow up with {attending_doctor} in 2 weeks, or sooner if symptoms worsen"
- "Return to see {attending_doctor} in 2 weeks for re-evaluation"

CRITICAL: The follow-up instruction MUST include "{attending_doctor}" - do not use any other doctor name.

Generate ONE realistic follow-up instruction (1-2 sentences, no quotes, just the instruction text):"""

        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.8, "num_predict": 100}
            )
            followup = response["message"]["content"].strip()
            followup = re.sub(r'^["\']|["\']$', '', followup)
            followup = followup.split('\n')[0].strip()
            if not followup:
                raise ValueError("Empty response")
            return followup
        except Exception as e:
            print(f"Warning: LLM failed for follow-up instructions: {e}, using template fallback")
            return f"Follow up with {attending_doctor} in 2 weeks"


class PHIGenerator:
    """Generate medical PHI (Protected Health Information) data with JSON tracking"""

    def __init__(self):
        self.current_patient = {}
        self.llm_generator = LLMPHIGenerator()
        # PHI tracking for JSON output
        self._filled_values = []
        self._pii_mappings = OrderedDict()

    def track_phi(self, value: str, phi_type: str):
        """Record a PHI value that was filled in.

        Args:
            value: The actual value filled (e.g., "Chen", "March 15, 2024")
            phi_type: The PHI type (e.g., "NAME", "DATE", "ID", "HOSPITAL", "FACILITY")
        """
        self._filled_values.append(value)
        if value not in self._pii_mappings:
            self._pii_mappings[value] = {"type": phi_type}

    @property
    def filled_values(self) -> List[str]:
        """Return ordered list of all filled PHI values (with duplicates)"""
        return list(self._filled_values)

    @property
    def pii_mappings(self) -> Dict:
        """Return mapping of unique PHI values to their types"""
        return dict(self._pii_mappings)

    def reset_patient(self):
        """Reset data for a new patient"""
        self.current_patient = {}
        self._filled_values = []
        self._pii_mappings = OrderedDict()

    def generate_patient_name(self, sex=None):
        """Generate patient name - 50% LLM, 50% Faker"""
        use_llm = USE_LLM and random.random() < LLM_PROBABILITY_PATIENT_NAME

        if use_llm:
            try:
                full_name = self.llm_generator.generate_patient_name_llm(sex)
                name_parts = full_name.split()
                if len(name_parts) >= 2:
                    first_name = name_parts[0]
                    last_name = " ".join(name_parts[1:])
                else:
                    raise ValueError("Name parsing failed")
            except Exception:
                first_name, last_name = self._faker_name(sex)
                full_name = f"{first_name} {last_name}"
        else:
            first_name, last_name = self._faker_name(sex)
            full_name = f"{first_name} {last_name}"

        self.current_patient['name'] = full_name
        self.current_patient['first_name'] = first_name
        self.current_patient['last_name'] = last_name
        return full_name

    def _faker_name(self, sex=None):
        """Helper to generate name with Faker"""
        if sex == 'F':
            return fake.first_name_female(), fake.last_name()
        elif sex == 'M':
            return fake.first_name_male(), fake.last_name()
        else:
            return fake.first_name(), fake.last_name()

    def generate_unit_no(self):
        """Generate unit number"""
        unit_no = str(random.randint(1000000, 9999999))
        self.current_patient['unit_no'] = unit_no
        return unit_no

    def generate_dates(self):
        """Generate admission and discharge dates"""
        admission_date = fake.date_between(start_date='-2y', end_date='-30d')
        stay_days = random.randint(1, 30)
        discharge_date = admission_date + timedelta(days=stay_days)

        admission_str = admission_date.strftime('%Y-%m-%d')
        discharge_str = discharge_date.strftime('%Y-%m-%d')

        self.current_patient['admission_date'] = admission_str
        self.current_patient['discharge_date'] = discharge_str

        return admission_str, discharge_str

    def generate_dob(self, sex=None):
        """Generate date of birth (18-90 years old)"""
        dob = fake.date_of_birth(minimum_age=18, maximum_age=90)
        dob_str = dob.strftime('%Y-%m-%d')
        self.current_patient['dob'] = dob_str
        return dob_str

    def generate_attending_name(self):
        """Generate attending physician name - 50% LLM, 50% Faker"""
        use_llm = USE_LLM and random.random() < LLM_PROBABILITY_DOCTOR_NAME

        if use_llm:
            try:
                attending = self.llm_generator.generate_doctor_name()
            except Exception:
                attending = f"Dr. {fake.first_name()} {fake.last_name()}"
        else:
            attending = f"Dr. {fake.first_name()} {fake.last_name()}"

        self.current_patient['attending'] = attending
        # Also store the name without "Dr. " prefix for tracking
        attending_name_only = attending.replace("Dr. ", "").strip()
        self.current_patient['attending_name_only'] = attending_name_only
        # Store last name separately for Dr. ___ references
        parts = attending_name_only.split()
        self.current_patient['attending_last_name'] = parts[-1] if parts else attending_name_only
        return attending

    def generate_facility_name(self):
        """Generate healthcare facility name"""
        use_llm = USE_LLM and random.random() < LLM_PROBABILITY_FACILITY

        if use_llm:
            try:
                facility = self.llm_generator.generate_facility_name()
            except Exception:
                facility = random.choice([
                    "Home Health Services", "Visiting Nurse Association",
                    "Community Health Services", "Home Care Partners"
                ])
        else:
            facility = random.choice([
                "Home Health Services", "Visiting Nurse Association",
                "Community Health Services", "Home Care Partners"
            ])

        self.current_patient['facility'] = facility
        return facility

    def generate_hospital_name(self):
        """Generate hospital name"""
        use_llm = USE_LLM and random.random() < LLM_PROBABILITY_HOSPITAL

        if use_llm:
            try:
                hospital = self.llm_generator.generate_hospital_name()
            except Exception:
                hospital = f"{fake.city()} {random.choice(['General Hospital', 'Medical Center', 'Regional Hospital'])}"
        else:
            hospital = f"{fake.city()} {random.choice(['General Hospital', 'Medical Center', 'Regional Hospital'])}"

        self.current_patient['hospital'] = hospital
        return hospital

    def generate_inline_date(self):
        """Generate a date in readable format for inline use (e.g., 'March 15, 2024')"""
        date = fake.date_between(start_date='-2y', end_date='today')
        return f"{date.strftime('%B')} {date.day}, {date.year}"

    def generate_doctor_last_name(self):
        """Generate just a doctor's last name for Dr. ___ references"""
        use_llm = USE_LLM and random.random() < LLM_PROBABILITY_DOCTOR_NAME
        if use_llm:
            try:
                full_name = self.llm_generator.generate_doctor_name()
                # Extract last name from "Dr. First Last"
                parts = full_name.replace("Dr. ", "").strip().split()
                return parts[-1] if parts else fake.last_name()
            except Exception:
                return fake.last_name()
        return fake.last_name()


def fill_phi_in_text(text, original_col2=""):
    """Fill all PHI placeholders in text and return JSON-compatible output.

    Returns:
        dict with keys:
            - original_text: the original text with ___ placeholders
            - filled_text: the text with PHI values filled in
            - pii_mappings: dict mapping each unique filled value to its PHI type
            - filled_values: ordered list of all filled values (preserves duplicates)
    """
    generator = PHIGenerator()
    original_text = text

    # Extract sex information (if available)
    sex = None
    sex_match = re.search(r'Sex:\s*([MF])', text)
    if sex_match:
        sex = sex_match.group(1)

    # =========================================================================
    # Process structured header fields (processed in typical text order)
    # =========================================================================

    # 1. Name: ___ (patient name)
    name_pattern = r'Name:\s*___'
    if re.search(name_pattern, text):
        patient_name = generator.generate_patient_name(sex)
        text = re.sub(name_pattern, f'Name:  {patient_name}', text, count=1)
        generator.track_phi(patient_name, "NAME")

    # 2. Unit No: ___ (medical record number)
    unit_pattern = r'Unit No:\s*___'
    if re.search(unit_pattern, text):
        unit_no = generator.generate_unit_no()
        text = re.sub(unit_pattern, f'Unit No:   {unit_no}', text, count=1)
        generator.track_phi(unit_no, "ID")

    # 3. Admission Date: ___ and Discharge Date: ___
    admission_pattern = r'Admission Date:\s*___'
    discharge_pattern = r'Discharge Date:\s*___'
    has_admission = bool(re.search(admission_pattern, text))
    has_discharge = bool(re.search(discharge_pattern, text))
    if has_admission or has_discharge:
        admission_date, discharge_date = generator.generate_dates()
        if has_admission:
            text = re.sub(admission_pattern, f'Admission Date:  {admission_date}', text)
            generator.track_phi(admission_date, "DATE")
        if has_discharge:
            text = re.sub(discharge_pattern, f'Discharge Date:   {discharge_date}', text)
            generator.track_phi(discharge_date, "DATE")

    # 4. Date of Birth: ___
    dob_pattern = r'Date of Birth:\s*___'
    if re.search(dob_pattern, text):
        dob = generator.generate_dob(sex)
        text = re.sub(dob_pattern, f'Date of Birth:  {dob}', text, count=1)
        generator.track_phi(dob, "DATE")

    # 5. Attending: ___ (attending physician)
    attending_pattern = r'Attending:\s*___\.?'
    if re.search(attending_pattern, text):
        attending = generator.generate_attending_name()
        text = re.sub(attending_pattern, f'Attending: {attending}', text)
        # Record without "Dr. " prefix as the PHI value
        attending_name_only = attending.replace("Dr. ", "").strip()
        generator.track_phi(attending_name_only, "NAME")

    # =========================================================================
    # Process inline patterns
    # =========================================================================

    # 6. Dear Mr./Ms. ___ (name in salutation)
    dear_pattern = r'Dear (Mr\.|Ms\.)\s*___,'
    dear_match = re.search(dear_pattern, text)
    if dear_match:
        title = dear_match.group(1)
        if generator.current_patient.get('last_name'):
            last_name = generator.current_patient['last_name']
        else:
            if title == 'Ms.':
                generator.generate_patient_name('F')
            else:
                generator.generate_patient_name('M')
            last_name = generator.current_patient['last_name']

        text = re.sub(dear_pattern, f'Dear {title} {last_name},', text)
        generator.track_phi(last_name, "NAME")

    # 6.5. Standalone Mr./Ms. ___ (not after Dear)
    standalone_mr_ms_pattern = r'(?<!Dear )(?<!Dear\s)(Mr\.|Ms\.)\s*___(?=[,\.:\n]|\s|$)'
    standalone_matches = list(re.finditer(standalone_mr_ms_pattern, text))
    if standalone_matches:
        if not generator.current_patient.get('last_name'):
            first_title = standalone_matches[0].group(1)
            if first_title == 'Ms.':
                generator.generate_patient_name('F')
            else:
                generator.generate_patient_name('M')
        last_name = generator.current_patient['last_name']

        # Track in forward order first, then replace from back to front
        for _ in standalone_matches:
            generator.track_phi(last_name, "NAME")

        for match in reversed(standalone_matches):
            title = match.group(1)
            match_start = match.start()
            match_end = match.end()

            if match_end < len(text):
                next_char = text[match_end]
                if next_char in ',.:':
                    replacement = f'{title} {last_name}{next_char}'
                    text = text[:match_start] + replacement + text[match_end + 1:]
                elif next_char in '\n\r':
                    replacement = f'{title} {last_name},'
                    text = text[:match_start] + replacement + text[match_end:]
                elif next_char in ' \t':
                    replacement = f'{title} {last_name},'
                    text = text[:match_start] + replacement + text[match_end:]
                else:
                    replacement = f'{title} {last_name},'
                    text = text[:match_start] + replacement + text[match_end:]
            else:
                replacement = f'{title} {last_name},'
                text = text[:match_start] + replacement

    # 7. Dr. ___ (doctor references in text body)
    #    Each Dr. ___ gets a unique last name for more realistic output.
    #    PHI recorded is just the last name (the "Dr." is structural text).
    dr_pattern = r'Dr\.\s*___'
    dr_matches = list(re.finditer(dr_pattern, text))
    doctor_last_names = []
    for i, match in enumerate(dr_matches):
        last_name = generator.generate_doctor_last_name()
        doctor_last_names.append(last_name)

    # Track in forward order
    for last_name in doctor_last_names:
        generator.track_phi(last_name, "NAME")

    # Replace from back to front to preserve positions
    for i, match in enumerate(reversed(dr_matches)):
        idx = len(dr_matches) - 1 - i
        last_name = doctor_last_names[idx]
        text = text[:match.start()] + f'Dr. {last_name}' + text[match.end():]

    # 8. Your ___ Team (hospital/team name)
    team_pattern = r'Your\s+___\s+Team'
    if re.search(team_pattern, text):
        hospital = generator.generate_hospital_name()
        text = re.sub(team_pattern, f'Your {hospital} Team', text)
        generator.track_phi(hospital, "HOSPITAL")

    # 9. Facility: ___ (healthcare facility)
    facility_pattern = r'Facility:\s*___'
    if re.search(facility_pattern, text):
        facility = generator.generate_facility_name()
        text = re.sub(facility_pattern, f'Facility:\n{facility}', text)
        generator.track_phi(facility, "FACILITY")

    # 10. at ___ (hospital name reference)
    at_hospital_pattern = r'at\s+___[\.\s]'
    if re.search(at_hospital_pattern, text):
        hospital = generator.generate_hospital_name()
        text = re.sub(at_hospital_pattern, f'at {hospital}. ', text)
        generator.track_phi(hospital, "HOSPITAL")

    # 11. ___ Dementia or other diagnosis starting with ___
    #     Diagnosis prefixes are clinical terms, NOT PHI - so we don't track them.
    diagnosis_start_pattern = r'^___\s+([A-Z][a-z]+)'
    diag_match = re.search(diagnosis_start_pattern, text, re.MULTILINE)
    if diag_match:
        diagnosis_prefixes = ["Lewy Body", "Vascular", "Alzheimer's", "Mixed", "Frontotemporal"]
        prefix = random.choice(diagnosis_prefixes)
        text = re.sub(diagnosis_start_pattern, f'{prefix} \\1', text, count=1)
        # NOT tracked as PHI (clinical terminology, not personal information)

    # =========================================================================
    # 12. Followup Instructions: ___ (SPECIAL HANDLING)
    #
    #     The LLM generates a full follow-up sentence, but NOT all of it is PHI.
    #     Only the doctor's name within the follow-up text is actual PHI.
    #     The rest (e.g., "Follow up in 2 weeks") is generic medical instructions.
    # =========================================================================
    followup_pattern = r'Followup Instructions:\s*___'
    if re.search(followup_pattern, text):
        # Ensure attending doctor has been generated
        attending = generator.current_patient.get('attending')
        if not attending:
            attending = generator.generate_attending_name()

        # Extract context for more realistic follow-up
        context_parts = []
        if 'Service:' in text:
            service_match = re.search(r'Service:\s*([^\n]+)', text)
            if service_match:
                context_parts.append(f"Service: {service_match.group(1).strip()}")
        if 'Discharge Diagnosis:' in text:
            diag_match = re.search(r'Discharge Diagnosis:\s*([^\n]+)', text)
            if diag_match:
                context_parts.append(f"Diagnosis: {diag_match.group(1).strip()}")
        patient_context = " | ".join(context_parts)

        # Generate follow-up text (100% LLM by default)
        use_llm = USE_LLM and random.random() < LLM_PROBABILITY_FOLLOWUP
        if use_llm:
            try:
                followup_text = generator.llm_generator.generate_followup_instructions(
                    attending, patient_context
                )
            except Exception:
                followup_text = f"Follow up with {attending} in 2 weeks"
        else:
            followup_text = f"Follow up with {attending} in 2 weeks"

        text = re.sub(followup_pattern, f'Followup Instructions:\n{followup_text}', text)

        # --- SPECIAL HANDLING: Extract only PHI from follow-up text ---
        # The attending doctor's name is PHI; the rest is not.
        attending_name_only = generator.current_patient.get(
            'attending_name_only',
            attending.replace("Dr. ", "").strip()
        )

        # Check if the full attending name (e.g., "Dr. John Smith") appears
        if attending in followup_text:
            generator.track_phi(attending_name_only, "NAME")
        elif attending_name_only in followup_text:
            generator.track_phi(attending_name_only, "NAME")

        # Also check for any other Dr. LastName patterns in the follow-up
        # (in case LLM introduced additional doctor names)
        dr_names_in_followup = re.findall(r'Dr\.\s+(\w+(?:\s+\w+)?)', followup_text)
        for name in dr_names_in_followup:
            # Skip if it's the attending (already tracked)
            if name != attending_name_only and name not in attending:
                generator.track_phi(name, "NAME")

        # Check for specific dates in follow-up (e.g., "March 15, 2024")
        # Note: relative times like "in 2 weeks" are NOT PHI
        specific_dates_in_followup = re.findall(
            r'\b(?:January|February|March|April|May|June|July|August|September|'
            r'October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            followup_text
        )
        for date_str in specific_dates_in_followup:
            generator.track_phi(date_str, "DATE")

    # =========================================================================
    # 13. Handle remaining ___ with context detection
    #
    #     For each remaining ___, check surrounding context to decide whether
    #     it should be filled with a date or a name.
    # =========================================================================
    remaining_pattern = r'___'
    while re.search(remaining_pattern, text):
        match = re.search(remaining_pattern, text)
        pos = match.start()

        # Get surrounding context
        context_before = text[max(0, pos - 60):pos].strip()
        context_after = text[pos + 3:min(len(text), pos + 60)].strip()

        # Get the last word before ___
        words_before = context_before.split()
        last_word = words_before[-1].lower().rstrip(',.:;') if words_before else ""

        # Determine replacement type based on context
        if last_word in DATE_CONTEXT_KEYWORDS:
            # Date context: fill with a readable date
            inline_date = generator.generate_inline_date()
            text = text[:pos] + inline_date + text[pos + 3:]
            generator.track_phi(inline_date, "DATE")

        elif context_before.rstrip().endswith(','):
            # After a comma (often in headers like "MRI WITHOUT CONTRAST, ___")
            # Likely a date
            inline_date = generator.generate_inline_date()
            text = text[:pos] + inline_date + text[pos + 3:]
            generator.track_phi(inline_date, "DATE")

        elif last_word in NAME_CONTEXT_KEYWORDS:
            # Name context: fill with a full name
            full_name = f"{fake.first_name()} {fake.last_name()}"
            text = text[:pos] + full_name + text[pos + 3:]
            generator.track_phi(full_name, "NAME")

        else:
            # Default: fill with a last name (most common for remaining ___)
            last_name = fake.last_name()
            text = text[:pos] + last_name + text[pos + 3:]
            generator.track_phi(last_name, "NAME")

    return {
        "original_text": original_text,
        "filled_text": text,
        "pii_mappings": generator.pii_mappings,
        "filled_values": generator.filled_values
    }


def process_csv_to_json(input_file, output_file, max_rows=None):
    """Process CSV file, fill PHI, and output JSON with PHI tracking.

    Args:
        input_file: Path to input CSV file (must have 'extracted_text' column)
        output_file: Path to output JSON file
        max_rows: Maximum number of rows to process (None = all rows)
    """
    print("=" * 60)
    print("PHI Data Filling Tool - JSON Output")
    print("=" * 60)
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"Using LLM: {USE_LLM} (Model: {MODEL_NAME})")
    if max_rows:
        print(f"Test mode: Processing only first {max_rows} non-empty rows")

    results = []
    rows_processed = 0

    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    infile = None
    encoding_used = None

    for encoding in encodings:
        try:
            infile = open(input_file, 'r', encoding=encoding, errors='replace', newline='')
            infile.seek(0)
            infile.readline()  # Test read
            infile.seek(0)
            encoding_used = encoding
            print(f"Opened file with {encoding} encoding")
            break
        except Exception:
            if infile:
                infile.close()
                infile = None
            continue

    if infile is None:
        raise ValueError(f"Could not read file {input_file} with any supported encoding")

    try:
        reader = csv.DictReader(infile)

        for row in reader:
            extracted_text = row.get('extracted_text', '')

            # Skip empty rows
            if not extracted_text or extracted_text.strip() == '':
                continue

            # Check row limit
            if max_rows and rows_processed >= max_rows:
                break

            original_col2 = row.get('original_col2', '')

            # Fill PHI and get JSON output
            result = fill_phi_in_text(extracted_text, original_col2)
            results.append(result)

            rows_processed += 1

            if rows_processed % 10 == 0:
                print(f"Processed {rows_processed} rows...")

    finally:
        if infile:
            infile.close()

    # Write JSON output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nCompleted! Total rows processed: {rows_processed}")
    print(f"Output saved to: {output_file}")

    # Print summary statistics
    total_phi = sum(len(r['filled_values']) for r in results)
    unique_phi = sum(len(r['pii_mappings']) for r in results)
    phi_types = {}
    for r in results:
        for val_info in r['pii_mappings'].values():
            t = val_info['type']
            phi_types[t] = phi_types.get(t, 0) + 1

    print(f"\n--- PHI Statistics ---")
    print(f"Total PHI values filled: {total_phi}")
    print(f"Unique PHI values: {unique_phi}")
    print(f"PHI types breakdown:")
    for phi_type, count in sorted(phi_types.items()):
        print(f"  {phi_type}: {count}")

    return results


if __name__ == "__main__":
    input_csv = "patient_extracted.csv"
    output_json = "patient_phi_filled.json"
    MAX_ROWS = None  # Set to None to process all rows, or a number for testing

    try:
        results = process_csv_to_json(input_csv, output_json, max_rows=MAX_ROWS)
        print("\n" + "=" * 60)
        print("SUCCESS! Completed successfully!")
        print("=" * 60)
        print(f"\nScript location: {__file__}")
        print(f"Output file: {output_json}")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
