import json
import random
import re
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    import keras
except ImportError:
    from tensorflow import keras
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from data_preprocessing import SRKRDataPreprocessor, clean_text, tokenize

# Intent tag to canonical knowledge base ID mapping (covers all 57 intent classes)
TAG_TO_KB_MAP = {
    "college_identity": "college_identity",
    "campus_size": "campus_overview",
    "campus_map": "campus_map_locations",
    "ug_programmes": "programmes_ug",
    "pg_programmes": "programmes_pg",
    "phd_programmes": "programmes_phd",
    "syllabus_directory": "syllabus_regulations",
    "regulation_help": "syllabus_lookup_requirements",
    "subject_syllabus": "syllabus_lookup_requirements",
    "model_papers": "model_papers",
    "open_electives": "open_electives_honours_minors",
    "honours_minors": "open_electives_honours_minors",
    "departments": "departments_and_programme_coordinators",
    "hostel": "hostel_overview",
    "hostel_contacts": "hostel_contacts",
    "library": "library",
    "transport": "transport",
    "transport_contact": "transport_contacts",
    "placements": "placements_overview",
    "placement_stats": "placement_statistics",
    "admissions": "admissions",
    "fees": "fees",
    "exams": "examinations",
    "academic_calendar": "academic_calendar",
    "research": "research",
    "clubs": "student_clubs",
    "sports": "sports",
    "wellbeing": "wellbeing",
    "anti_ragging": "ragging",
    "ece_labs": "ece_labs",
    "general_contact": "college_contact_channels",
    "fallback": "general_unknown",
    "accreditation": "accreditation",
    "research_infrastructure": "research_infrastructure",
    "mechanical_department": "mechanical_department_profile",
    "mechanical_labs": "mechanical_labs",
    "leadership": "leadership_executive",
    "hod_contacts": "hod_contacts",
    "programme_coordinator_contacts": "programme_coordinator_contacts",
    "technical_clubs": "student_clubs_technical",
    "creative_clubs": "student_clubs_creative",
    "service_clubs": "student_clubs_service",
    "sports_clubs": "sports_clubs",
    "samagra": "samagra_festival",
    "jaitra": "jaitra_sports_carnival",
    "hackathons": "hackathons_project_expos",
    "hostel_safety": "hostel_rules_and_support",
    "library_resources": "library_resources",
    "library_services": "library_services",
    "r23_syllabus": "r23_btech_catalogue",
    "r20_syllabus": "r20_btech_catalogue",
    "r26_syllabus": "r26_first_year_catalogue",
    "r24_bba": "r24_bba_catalogue",
    "r25_mtech": "r25_mtech_catalogue",
    "placement_contact": "placement_contact_team"
}

# Pre-defined natural conversational answers for short keywords and common queries
# Grounded strictly in SRKR_detailed_chatbot_dataset_v2.json and official sources
SHORT_QUERY_RESPONSES = {
    "map": (
        "SRKR Engineering College has several mapped campus landmarks, including N Block, "
        "Silver Jubilee Block, Civil Department, WET Centre, Ground/Pitch and SRKR Pond.\n\n"
        "You can view these locations on the public campus map: https://mapcarta.com/W851081636\n\n"
        "Note: This public map does not provide a complete indoor map or exact classroom locations."
    ),
    "campus map": (
        "SRKR Engineering College has several mapped campus landmarks, including N Block, "
        "Silver Jubilee Block, Civil Department, WET Centre, Ground/Pitch and SRKR Pond.\n\n"
        "You can view these locations on the public campus map: https://mapcarta.com/W851081636\n\n"
        "Note: This public map does not provide a complete indoor map or exact classroom locations."
    ),
    "hostel": (
        "SRKR Engineering College provides separate on-campus hostel accommodation for both boys and girls. "
        "Resident wardens attend to students and provide medical and essential assistance. "
        "Hostel residents have access to the gymnasium, sports playground, and indoor gaming facilities "
        "under a strict no-ragging ethos.\n\n"
        "**Hostel Office Contacts:**\n"
        "- Chief Warden: 8074874235\n"
        "- Boys Hostel: 9492123311\n"
        "- Girls Hostel: 7995383711\n"
        "- Email: hostels@srkrec.ac.in\n\n"
        "Note: Room availability, fees, rules, mess timings, and admission procedures should be confirmed directly with the hostel office.\n\n"
        "Official Hostels Page: https://www.srkrec.ac.in/hostels/"
    ),
    "hostels": (
        "SRKR Engineering College provides separate on-campus hostel accommodation for both boys and girls. "
        "Resident wardens attend to students and provide medical and essential assistance. "
        "Hostel residents have access to the gymnasium, sports playground, and indoor gaming facilities "
        "under a strict no-ragging ethos.\n\n"
        "**Hostel Office Contacts:**\n"
        "- Chief Warden: 8074874235\n"
        "- Boys Hostel: 9492123311\n"
        "- Girls Hostel: 7995383711\n"
        "- Email: hostels@srkrec.ac.in\n\n"
        "Note: Room availability, fees, rules, mess timings, and admission procedures should be confirmed directly with the hostel office.\n\n"
        "Official Hostels Page: https://www.srkrec.ac.in/hostels/"
    ),
    "placements": (
        "SRKR Engineering College has an active Training & Placement (T&P) team that publishes batch-specific placement statistics.\n\n"
        "For the **2025-26 reporting period**, the published placement figures include:\n"
        "- **87.3% placements**\n"
        "- **19.00 LPA top package** (attributed to AT&T)\n"
        "- **79 on-campus recruiters**\n"
        "- **15 industry partners**\n"
        "- (Homepage snapshot also records 2,344+ placement offers for the 2025-26 batch)\n\n"
        "**Training & Placement Office:**\n"
        "- Location: U Block, SRKR Engineering College, Bhimavaram - 534204\n"
        "- Office Hours: Monday - Saturday, 9:00 AM - 5:00 PM\n"
        "- Dean of Placements & Industry Relations: Dr. K.R. Satyanarayana (krsn@srkrec.ac.in / 9704195376)\n"
        "- General Placement Email: dean.placements@srkrec.ac.in\n"
        "- Manager Corporate Relations: G. Ashok Reddy (corporaterelations@srkrec.ac.in / 9440359517)\n"
        "- Manager International Affairs: Baskar C (baskar.c@srkrec.ac.in / 9790997707)\n\n"
        "Official Placements Team: https://www.srkrec.ac.in/placements-team/"
    ),
    "placement": (
        "SRKR Engineering College has an active Training & Placement (T&P) team that publishes batch-specific placement statistics.\n\n"
        "For the **2025-26 reporting period**, the published placement figures include:\n"
        "- **87.3% placements**\n"
        "- **19.00 LPA top package** (attributed to AT&T)\n"
        "- **79 on-campus recruiters**\n"
        "- **15 industry partners**\n"
        "- (Homepage snapshot also records 2,344+ placement offers for the 2025-26 batch)\n\n"
        "**Training & Placement Office:**\n"
        "- Location: U Block, SRKR Engineering College, Bhimavaram - 534204\n"
        "- Office Hours: Monday - Saturday, 9:00 AM - 5:00 PM\n"
        "- Dean of Placements & Industry Relations: Dr. K.R. Satyanarayana (krsn@srkrec.ac.in / 9704195376)\n"
        "- General Placement Email: dean.placements@srkrec.ac.in\n"
        "- Manager Corporate Relations: G. Ashok Reddy (corporaterelations@srkrec.ac.in / 9440359517)\n"
        "- Manager International Affairs: Baskar C (baskar.c@srkrec.ac.in / 9790997707)\n\n"
        "Official Placements Team: https://www.srkrec.ac.in/placements-team/"
    ),
    "fees": (
        "SRKR Engineering College publishes its fee structure for tuition, university/examination charges, "
        "hostel, and transport on the official website. Because fee amounts and regulatory guidelines depend on the "
        "admission year, category, and programme, please refer to the latest published fee breakdown:\n\n"
        "Official Fee Structure: https://www.srkrec.ac.in/fee-structure/\n\n"
        "Note: The chatbot does not estimate or quote older fees as current. For verified category-wise fee information, "
        "please contact the SRKR Admissions Office (+91 9848823332 / info@srkrec.ac.in)."
    ),
    "fee": (
        "SRKR Engineering College publishes its fee structure for tuition, university/examination charges, "
        "hostel, and transport on the official website. Because fee amounts and regulatory guidelines depend on the "
        "admission year, category, and programme, please refer to the latest published fee breakdown:\n\n"
        "Official Fee Structure: https://www.srkrec.ac.in/fee-structure/\n\n"
        "Note: The chatbot does not estimate or quote older fees as current. For verified category-wise fee information, "
        "please contact the SRKR Admissions Office (+91 9848823332 / info@srkrec.ac.in)."
    ),
    "library": (
        "The Sri SRK Raju Central Library spans over 4,160 square metres and is fully automated with RFID technology. "
        "It houses more than 24,000 titles, about 1.19 lakh (1,19,000) volumes, 6,458 e-journals, and 110 print journals.\n\n"
        "**Key Facilities & Resources:**\n"
        "- Digital Learning Centre with 252 systems and e-learning resources\n"
        "- Reading room, book bank, and reference section\n"
        "- E-resources & databases: ScienceDirect, ASCE Library, IEEE Xplore, Springer, DELNET, J-Gate, N-LIST, NPTEL, and Knimbus\n"
        "- Online Public Access Catalogue (OPAC) for real-time asset search\n\n"
        "Official Library Portal: https://www.srkrec.ac.in/library/"
    ),
    "aiml": (
        "SRKR Engineering College offers specialized undergraduate B.Tech programmes in artificial intelligence:\n"
        "- **B.Tech in Artificial Intelligence and Machine Learning (AI & ML)**\n"
        "- **B.Tech in Artificial Intelligence and Data Science (AI & Data Science)**\n\n"
        "These programmes are supported by dedicated programme coordinators (AI & ML: pc.aiml@srkrec.ac.in; AI & Data Science: hod.aids@srkrec.ac.in) "
        "and campus research infrastructure including recognized research centres, the AICTE IDEA Lab, and DST-FIST funding.\n\n"
        "Academic Programmes Directory: https://www.srkrec.ac.in/programmes/"
    ),
    "ai": (
        "SRKR Engineering College offers specialized undergraduate B.Tech programmes in artificial intelligence:\n"
        "- **B.Tech in Artificial Intelligence and Machine Learning (AI & ML)**\n"
        "- **B.Tech in Artificial Intelligence and Data Science (AI & Data Science)**\n\n"
        "These programmes are supported by dedicated programme coordinators (AI & ML: pc.aiml@srkrec.ac.in; AI & Data Science: hod.aids@srkrec.ac.in) "
        "and campus research infrastructure including recognized research centres, the AICTE IDEA Lab, and DST-FIST funding.\n\n"
        "Academic Programmes Directory: https://www.srkrec.ac.in/programmes/"
    ),
    "courses": (
        "SRKR Engineering College (Autonomous) offers degree programmes across undergraduate, postgraduate, and doctoral levels:\n\n"
        "- **Undergraduate (B.Tech & BBA):** AI & Data Science, AI & ML, Civil Engineering, CS & Business System, CS & Design, CSE, ECE, EEE, IT, Mechanical Engineering, CIC (CSE-IoT & Cyber Security), CS & IT, and Bachelor of Business Administration (BBA).\n"
        "- **Postgraduate (M.Tech):** Specializations in CAD/CAM, Structural Engineering, Communication Systems, Computer Science and Technology, Information Technology, Power Systems and Automation, Electric Vehicle Technology, and Artificial Intelligence.\n"
        "- **Doctoral (Ph.D.):** Research areas in Civil Engineering, CSE, ECE, EEE, IT, Mechanical Engineering, Mathematics, Chemistry, Physics, and English.\n\n"
        "Full Programme Details: https://www.srkrec.ac.in/programmes/"
    ),
    "branches": (
        "SRKR Engineering College (Autonomous) offers undergraduate B.Tech programmes across multiple engineering and computing branches:\n\n"
        "- **Computing & Emerging Technologies:** CSE, AI & ML, AI & Data Science, CS & Business System, CS & Design, CS & IT, IT, and CIC (IoT & Cyber Security).\n"
        "- **Circuit Engineering:** Electronics & Communication Engineering (ECE), Electrical & Electronics Engineering (EEE).\n"
        "- **Core Engineering:** Mechanical Engineering, Civil Engineering.\n"
        "- **Management:** Bachelor of Business Administration (BBA).\n\n"
        "Official Programmes Directory: https://www.srkrec.ac.in/programmes/"
    ),
    "programmes": (
        "SRKR Engineering College offers programmes at undergraduate (B.Tech, BBA), postgraduate (M.Tech), "
        "and doctoral (Ph.D.) levels across disciplines including Computer Science, AI, Electronics, Electrical, Mechanical, Civil Engineering, and Basic Sciences.\n\n"
        "Official Programmes Directory: https://www.srkrec.ac.in/programmes/"
    ),
    "transport": (
        "SRKR operates a dedicated college bus fleet connecting students and staff commuting from Bhimavaram and surrounding rural and urban areas. "
        "The buses cover multiple routes and aim to reach campus at least 15 minutes before classes and office hours.\n\n"
        "**Transport Office Contact:**\n"
        "- Location: Admin Block, SRKR Engineering College, Bhimavaram\n"
        "- Phone: +91 884 222 6666\n"
        "- Email: transport@srkrec.ac.in\n\n"
        "Note: A static route/time table is not published in the extracted dataset; please contact the Transport Section for current routes, stops, seat availability, fees, and timings.\n\n"
        "Official Transportation Page: https://www.srkrec.ac.in/transportation/"
    ),
    "bus": (
        "SRKR operates a dedicated college bus fleet connecting students and staff commuting from Bhimavaram and surrounding rural and urban areas. "
        "The buses cover multiple routes and aim to reach campus at least 15 minutes before classes and office hours.\n\n"
        "**Transport Office Contact:**\n"
        "- Location: Admin Block, SRKR Engineering College, Bhimavaram\n"
        "- Phone: +91 884 222 6666\n"
        "- Email: transport@srkrec.ac.in\n\n"
        "Note: A static route/time table is not published in the extracted dataset; please contact the Transport Section for current routes, stops, seat availability, fees, and timings.\n\n"
        "Official Transportation Page: https://www.srkrec.ac.in/transportation/"
    ),
    "contact": (
        "You can contact SRKR Engineering College through the following official channels:\n\n"
        "- **Institution:** Sagi Rama Krishnam Raju Engineering College (SRKR Engineering College)\n"
        "- **Campus Address:** SRKR Marg, China Amiram, Bhimavaram, West Godavari District, Andhra Pradesh, India - 534204\n"
        "- **General Phone:** +91 9848823332\n"
        "- **General Email:** info@srkrec.ac.in\n"
        "- **Official Website:** https://www.srkrec.ac.in\n\n"
        "**Dedicated Section Contacts:**\n"
        "- Hostel Office: 8074874235 / hostels@srkrec.ac.in\n"
        "- Transport Section: +91 884 222 6666 / transport@srkrec.ac.in\n"
        "- Training & Placement Cell: 9704195376 / dean.placements@srkrec.ac.in"
    ),
    "departments": (
        "SRKR Engineering College (Autonomous) comprises multiple academic and basic science departments:\n\n"
        "- **Engineering Departments:** Civil (civil@srkrec.ac.in), CSE (cse@srkrec.ac.in), ECE (ece@srkrec.ac.in), EEE (eee@srkrec.ac.in), Mechanical (mech@srkrec.ac.in), and IT (it@srkrec.ac.in).\n"
        "- **Specialized Programme Coordination:** CS & IT (csit@srkrec.ac.in), AI & ML (pc.aiml@srkrec.ac.in), AI & Data Science (hod.aids@srkrec.ac.in), CS & Business System (hod.csbs@srkrec.ac.in), CSE-IoT & Cyber Security (hod.cic@srkrec.ac.in), CS & Design (hod.csd@srkrec.ac.in).\n"
        "- **Management & Basic Sciences:** Management Studies (hod.bba@srkrec.ac.in), Mathematics (maths@srkrec.ac.in), Chemistry (chemistry@srkrec.ac.in), Physics (physics@srkrec.ac.in), and English (english@srkrec.ac.in).\n\n"
        "Complete Leadership & Faculty Directory: https://www.srkrec.ac.in/leadership/"
    ),
    "department": (
        "SRKR Engineering College (Autonomous) comprises multiple academic and basic science departments:\n\n"
        "- **Engineering Departments:** Civil (civil@srkrec.ac.in), CSE (cse@srkrec.ac.in), ECE (ece@srkrec.ac.in), EEE (eee@srkrec.ac.in), Mechanical (mech@srkrec.ac.in), and IT (it@srkrec.ac.in).\n"
        "- **Specialized Programme Coordination:** CS & IT (csit@srkrec.ac.in), AI & ML (pc.aiml@srkrec.ac.in), AI & Data Science (hod.aids@srkrec.ac.in), CS & Business System (hod.csbs@srkrec.ac.in), CSE-IoT & Cyber Security (hod.cic@srkrec.ac.in), CS & Design (hod.csd@srkrec.ac.in).\n"
        "- **Management & Basic Sciences:** Management Studies (hod.bba@srkrec.ac.in), Mathematics (maths@srkrec.ac.in), Chemistry (chemistry@srkrec.ac.in), Physics (physics@srkrec.ac.in), and English (english@srkrec.ac.in).\n\n"
        "Complete Leadership & Faculty Directory: https://www.srkrec.ac.in/leadership/"
    ),
    "scholarships": (
        "Specific scholarship schemes and government welfare policies (such as state fee reimbursement programs) "
        "are not currently catalogued in the chatbot's verified knowledge base.\n\n"
        "Students seeking scholarship details, eligibility guidelines, or application assistance should "
        "consult the SRKR Administrative Office directly (+91 9848823332 / info@srkrec.ac.in) or visit the official fee structure page:\n"
        "https://www.srkrec.ac.in/fee-structure/"
    ),
    "scholarship": (
        "Specific scholarship schemes and government welfare policies (such as state fee reimbursement programs) "
        "are not currently catalogued in the chatbot's verified knowledge base.\n\n"
        "Students seeking scholarship details, eligibility guidelines, or application assistance should "
        "consult the SRKR Administrative Office directly (+91 9848823332 / info@srkrec.ac.in) or visit the official fee structure page:\n"
        "https://www.srkrec.ac.in/fee-structure/"
    ),
    "syllabus": (
        "SRKR's official syllabus directory is organized by academic regulation and branch:\n\n"
        "- R26: B.Tech first-year syllabus and model papers across branches\n"
        "- R25: M.Tech specialization syllabus and model papers\n"
        "- R24: BBA first, second, and third year\n"
        "- R23: B.Tech first through fourth year across branches, open electives, Honours, and Minors\n"
        "- R20: B.Tech year-wise resources\n\n"
        "To find your exact syllabus, please specify your degree, branch, and regulation year, or browse the directory directly:\n"
        "Official Syllabus Portal: https://www.srkrec.ac.in/syllabus/"
    ),
    "admission": (
        "Admissions to SRKR Engineering College (Autonomous) are conducted through official statutory channels:\n\n"
        "- **Undergraduate (B.Tech):** Admitted via AP EAPCET state centralized counselling alongside official seat categories.\n"
        "- **Lateral Entry (Direct 2nd Year):** Through AP ECET for polytechnic diploma holders.\n"
        "- **Postgraduate:** M.Tech via GATE / AP PGECET; MCA via AP ICET.\n\n"
        "Because eligibility, seat categories, counselling rounds, documents, and cut-off ranks are academic-year dependent, "
        "please consult the live admission sections:\n"
        "- Admission Overview: https://www.srkrec.ac.in/admission-overview/\n"
        "- Admission Procedure: https://www.srkrec.ac.in/admission-procedure/\n"
        "- Fee Structure: https://www.srkrec.ac.in/fee-structure/\n\n"
        "For direct admission inquiries, contact the college office at +91 9848823332 / info@srkrec.ac.in."
    ),
    "admissions": (
        "Admissions to SRKR Engineering College (Autonomous) are conducted through official statutory channels:\n\n"
        "- **Undergraduate (B.Tech):** Admitted via AP EAPCET state centralized counselling alongside official seat categories.\n"
        "- **Lateral Entry (Direct 2nd Year):** Through AP ECET for polytechnic diploma holders.\n"
        "- **Postgraduate:** M.Tech via GATE / AP PGECET; MCA via AP ICET.\n\n"
        "Because eligibility, seat categories, counselling rounds, documents, and cut-off ranks are academic-year dependent, "
        "please consult the live admission sections:\n"
        "- Admission Overview: https://www.srkrec.ac.in/admission-overview/\n"
        "- Admission Procedure: https://www.srkrec.ac.in/admission-procedure/\n"
        "- Fee Structure: https://www.srkrec.ac.in/fee-structure/\n\n"
        "For direct admission inquiries, contact the college office at +91 9848823332 / info@srkrec.ac.in."
    )
}

# Refined natural conversational answers for meta-instruction knowledge base topics
# Grounded strictly in SRKR_detailed_chatbot_dataset_v2.json
KB_REFINED_ANSWERS = {
    "college_identity": (
        "**Sagi Rama Krishnam Raju (SRKR) Engineering College** is an autonomous engineering institution established in 1980.\n\n"
        "- **Location:** SRKR Marg, China Amiram, Bhimavaram, West Godavari District, Andhra Pradesh - 534204\n"
        "- **Campus Size:** 30 Acres\n"
        "- **Affiliation:** Autonomous institution affiliated to JNTU Kakinada (JNTUK)\n"
        "- **Accreditation:** NAAC with 'A+' Grade (CGPA 3.42, Cycle II); NBA accredited engineering programmes\n"
        "- **Contact:** +91 9848823332 | info@srkrec.ac.in | https://www.srkrec.ac.in"
    ),
    "leadership_executive": (
        "The executive leadership of SRKR Engineering College (Autonomous) includes:\n\n"
        "- **Director:** Dr. M. Jagapathi Raju\n"
        "- **Principal:** Dr. K.V. Murali Krishna Raju\n"
        "- **Chief Administrative Officer (CAO):** Mr. Ch. Dileep Chakravarthy\n\n"
        "**Key Academic Deans:**\n"
        "- Dean Academics: Dr. V. Chandra Sekhar\n"
        "- Dean R&D: Dr. T. Vamsi Nagaraju\n"
        "- Dean Placements & Industry Relations: Dr. K.R. Satyanarayana\n"
        "- Dean IQAC: Dr. S. Rajesh\n"
        "- Dean Student Affairs & Alumni: Dr. G. Harish Kumar Varma\n"
        "- Dean Applied Sciences: Dr. B.S. Diwakar\n\n"
        "Official Leadership Directory: https://www.srkrec.ac.in/leadership/"
    ),
    "admissions": (
        "Admissions to SRKR Engineering College (Autonomous) are conducted through official statutory channels:\n\n"
        "- **Undergraduate (B.Tech):** Allotted via AP EAPCET state centralized counselling alongside official seat categories.\n"
        "- **Lateral Entry (Direct 2nd Year):** For polytechnic diploma holders via AP ECET.\n"
        "- **Postgraduate:** M.Tech through GATE / AP PGECET, and MCA through AP ICET.\n\n"
        "Because eligibility criteria, seat categories, counselling rounds, documents, and cut-offs change by academic year, "
        "please refer to the official admissions portal:\n"
        "Official Admissions Overview: https://www.srkrec.ac.in/admission-overview/\n"
        "Official Admission Procedure: https://www.srkrec.ac.in/admission-procedure/"
    ),
    "admissions_contact_path": (
        "For verified admission eligibility, seat categories, counselling rounds, brochures, and fee structures, "
        "consult the official SRKR admissions sections or contact the college office:\n\n"
        "- Phone: +91 9848823332\n"
        "- Email: info@srkrec.ac.in\n"
        "- Address: Admin Block, SRKR Engineering College, Bhimavaram - 534204\n\n"
        "Official Admissions Page: https://www.srkrec.ac.in/admission-overview/"
    ),
    "examinations": (
        "SRKR Engineering College operates a dedicated Autonomous Examinations Cell with an online portal (srkrexams.in).\n\n"
        "Students should consult the official portal for verified examination notices, timetables, results, and services:\n"
        "- End-semester examination timetables and schedules\n"
        "- Regular and supplementary examination fee notifications\n"
        "- Grade cards, SGPA/CGPA evaluation, and results\n"
        "- Revaluation and scripts verification services\n\n"
        "Official Examinations Portal: https://srkrexams.in"
    ),
    "exams": (
        "SRKR Engineering College operates a dedicated Autonomous Examinations Cell with an online portal (srkrexams.in).\n\n"
        "Students should consult the official portal for verified examination notices, timetables, results, and services:\n"
        "- End-semester examination timetables and schedules\n"
        "- Regular and supplementary examination fee notifications\n"
        "- Grade cards, SGPA/CGPA evaluation, and results\n"
        "- Revaluation and scripts verification services\n\n"
        "Official Examinations Portal: https://srkrexams.in"
    ),
    "hostel_overview": (
        "SRKR Engineering College provides separate on-campus hostel accommodation for both boys and girls. "
        "Resident wardens attend to students and provide medical and essential assistance. "
        "Hostel residents have access to the gymnasium, sports playground, and indoor gaming facilities "
        "under a strict no-ragging ethos.\n\n"
        "**Hostel Office Contacts:**\n"
        "- Chief Warden: 8074874235\n"
        "- Boys Hostel: 9492123311\n"
        "- Girls Hostel: 7995383711\n"
        "- Email: hostels@srkrec.ac.in\n\n"
        "Official Hostels Page: https://www.srkrec.ac.in/hostels/"
    ),
    "hostel_contacts": (
        "**SRKR Hostel Office Contact Details:**\n\n"
        "- **Chief Warden:** 8074874235\n"
        "- **Boys Hostel Warden:** 9492123311\n"
        "- **Girls Hostel Warden:** 7995383711\n"
        "- **Hostel Office Email:** hostels@srkrec.ac.in\n"
        "- **Location:** On-campus hostel blocks, SRKR Engineering College, Bhimavaram\n\n"
        "Official Hostels Page: https://www.srkrec.ac.in/hostels/"
    ),
    "placements_overview": (
        "SRKR Engineering College has an active Training & Placement (T&P) team that publishes batch-specific placement statistics.\n\n"
        "For the **2025-26 reporting period**, published placement figures include:\n"
        "- **87.3% placements**\n"
        "- **19.00 LPA top package** (attributed to AT&T)\n"
        "- **79 on-campus recruiters**\n"
        "- **15 industry partners**\n"
        "- (2,344+ placement offers recorded for the 2025-26 batch)\n\n"
        "**Training & Placement Office:**\n"
        "- Location: U Block, SRKR Engineering College, Bhimavaram - 534204\n"
        "- Dean of Placements: Dr. K.R. Satyanarayana (krsn@srkrec.ac.in / 9704195376)\n"
        "- General Placement Email: dean.placements@srkrec.ac.in\n\n"
        "Official Placements Team: https://www.srkrec.ac.in/placements-team/"
    ),
    "placement_statistics": (
        "For the **2025-26 reporting period**, SRKR Engineering College recorded the following published placement figures:\n\n"
        "- **Placement Percentage:** 87.3%\n"
        "- **Highest Package:** 19.00 LPA (attributed to AT&T)\n"
        "- **Recruiters On-Campus:** 79\n"
        "- **Industry Partners:** 15\n"
        "- **Batch Placement Offers:** 2,344+ offers recorded on the homepage snapshot\n\n"
        "Dean of Placements: Dr. K.R. Satyanarayana (9704195376 / krsn@srkrec.ac.in)\n"
        "Official Placements Portal: https://www.srkrec.ac.in/placements-team/"
    ),
    "placement_contact_team": (
        "**SRKR Training & Placement Office Contacts:**\n\n"
        "- **Location:** U Block, SRKR Engineering College, Bhimavaram - 534204\n"
        "- **Dean of Placements & Industry Relations:** Dr. K.R. Satyanarayana (9704195376 / krsn@srkrec.ac.in)\n"
        "- **Manager Corporate Relations:** G. Ashok Reddy (9440359517 / corporaterelations@srkrec.ac.in)\n"
        "- **Manager International Affairs:** Baskar C (9790997707 / baskar.c@srkrec.ac.in)\n"
        "- **General Placement Inquiries:** dean.placements@srkrec.ac.in\n\n"
        "Official Placements Page: https://www.srkrec.ac.in/placements-team/"
    ),
    "student_clubs": (
        "SRKR supports an active student campus life through a diverse range of official student clubs:\n\n"
        "- **Technical & Professional Groups:** E-Mobility, Drone, Embedded & IoT, VLSI, AI, Networking and Cybersecurity, Quantum Technologies, Prompt Engineering, Deep Technology, Web Development, Coding, IEEE, ACM, CSI, IEI, ISTE, IGBC, IIPE, and IETE.\n"
        "- **Creative & Cultural Groups:** Content & Media, Fine Arts, Dance, LOLO (music), Theatre, Cine Club, Language Nest, Toastmasters, and IKS (Indian Knowledge Systems).\n"
        "- **Service & Community Clubs:** NSS (National Service Scheme), Women Empowerment Cell, and Unnat Bharat Abhiyan (UBA).\n"
        "- **Sports:** Badminton, Cricket, Football, Volleyball, Basketball, Kabaddi, and Athletics.\n\n"
        "Official Student Clubs Directory: https://www.srkrec.ac.in/student-clubs/"
    ),
    "student_clubs_technical": (
        "SRKR hosts an extensive network of specialized technical clubs and professional society chapters:\n\n"
        "- **Emerging Tech Clubs:** AI Club, Deep Technology, Prompt Engineering, Quantum Technologies, Embedded & IoT Club, VLSI Club, Drone Club, E-Mobility, Web Development, and Coding Club.\n"
        "- **Professional Society Chapters:** IEEE Student Branch, ACM, Computer Society of India (CSI), IEI, ISTE, IGBC, IIPE, and IETE.\n\n"
        "Official Student Clubs: https://www.srkrec.ac.in/student-clubs/"
    ),
    "student_clubs_creative": (
        "SRKR supports creative and cultural expression through student-led clubs:\n\n"
        "- **Content & Media Club**\n"
        "- **Fine Arts Club**\n"
        "- **Dance Club**\n"
        "- **LOLO (Music Club)**\n"
        "- **Theatre & Cine Club**\n"
        "- **Language Nest & Toastmasters** (Public speaking and communication)\n"
        "- **Indian Knowledge Systems (IKS)**\n\n"
        "Official Student Clubs: https://www.srkrec.ac.in/student-clubs/"
    ),
    "student_clubs_service": (
        "SRKR students lead community and social development through dedicated groups:\n\n"
        "- **National Service Scheme (NSS):** Community outreach, blood donation drives, health awareness camps, and local service.\n"
        "- **Women Empowerment Cell:** Leadership programmes, student safety, and welfare initiatives.\n"
        "- **Unnat Bharat Abhiyan (UBA):** Village adoption and grassroots rural technology applications.\n\n"
        "Official Student Clubs: https://www.srkrec.ac.in/student-clubs/"
    ),
    "samagra_festival": (
        "**SAMAGRA** is SRKR Engineering College's 3-day flagship annual techno-cultural festival. "
        "It features national-level technical competitions across multidisciplinary emerging areas, research conclaves, "
        "student project expos, creative cultural events, and annual-day celebrations.\n\n"
        "Official Student Clubs & Events Page: https://www.srkrec.ac.in/student-clubs/"
    ),
    "jaitra_sports_carnival": (
        "**JAITRA** is SRKR Engineering College's annual national-level sports carnival conducted every year in January. "
        "It features inter-collegiate competitions in athletics, badminton, cricket, basketball, football, volleyball, and kabaddi.\n\n"
        "Official Sports & Activities: https://www.srkrec.ac.in/student-clubs/"
    ),
    "hackathons_project_expos": (
        "SRKR Engineering College regularly conducts departmental hackathons, ideathons, coding sprints, "
        "and project expos throughout the academic year. These events provide students with hands-on problem solving "
        "in AI, IoT, sustainability, robotics, and software development.\n\n"
        "Official Events & Activities: https://www.srkrec.ac.in/student-clubs/"
    ),
    "ragging": (
        "SRKR Engineering College enforces a strict, zero-tolerance **Anti-Ragging Policy** under a firm no-ragging ethos across campus and hostels:\n\n"
        "- Ragging in any form is prohibited under statutory guidelines.\n"
        "- The institution maintains active Anti-Ragging Committees and monitoring squads.\n"
        "- Resident wardens provide 24/7 student care and assistance.\n\n"
        "To report any concerns, contact the college office directly (+91 9848823332 / info@srkrec.ac.in) or the hostel chief warden (8074874235).\n\n"
        "Official Hostels Page: https://www.srkrec.ac.in/hostels/"
    ),
    "ece_labs": (
        "The **Department of Electronics and Communication Engineering (ECE)** operates eleven specialized laboratories, including:\n\n"
        "- Electronic Devices & Circuits Lab\n"
        "- Digital Signal Processing (DSP) & Microprocessors Lab\n"
        "- Microwave & Optical Communications Lab\n"
        "- VLSI Design Lab\n"
        "- Embedded Systems & IoT Lab\n"
        "- Communication Systems Lab\n\n"
        "These labs support practical laboratory coursework, research projects, and student innovation.\n\n"
        "Official ECE Facilities: https://www.srkrec.ac.in/facilities-ece/"
    ),
    "mechanical_department_profile": (
        "The **Department of Mechanical Engineering** was established in 1980 as one of SRKR's founding departments.\n\n"
        "- **Programmes:** B.Tech in Mechanical Engineering, M.Tech in CAD/CAM, and Ph.D. research.\n"
        "- **Accreditation:** NBA accredited.\n"
        "- **Laboratories:** CAD/CAM, Robotics, Thermal Engineering, Dynamics, and Manufacturing Technology.\n"
        "- **Contact:** mech@srkrec.ac.in\n\n"
        "Official Department Page: https://www.srkrec.ac.in/mechanical-engineering/"
    ),
    "mechanical_labs": (
        "The Mechanical Engineering department maintains specialized experimental and computing laboratories including:\n\n"
        "- CAD/CAM & Simulation Lab\n"
        "- Robotics & Automation Lab\n"
        "- Thermal Engineering & Heat Transfer Lab\n"
        "- Dynamics & Kinematics of Machinery Lab\n"
        "- Manufacturing & Machine Tools Lab\n"
        "- Metallurgy & Materials Testing Lab\n\n"
        "Official Facilities Page: https://www.srkrec.ac.in/mechanical-engineering/"
    ),
    "transport": (
        "SRKR operates a dedicated college bus fleet connecting students and staff commuting from Bhimavaram and surrounding rural and urban areas. "
        "The buses cover multiple routes and aim to reach campus at least 15 minutes before classes and office hours.\n\n"
        "**Transport Office Contact:**\n"
        "- Location: Admin Block, SRKR Engineering College, Bhimavaram\n"
        "- Phone: +91 884 222 6666\n"
        "- Email: transport@srkrec.ac.in\n\n"
        "Official Transportation Page: https://www.srkrec.ac.in/transportation/"
    ),
    "wellbeing": (
        "SRKR Engineering College provides student support and welfare through designated campus facilities:\n\n"
        "- The official website maintains a dedicated Well-Being section for personal and academic guidance.\n"
        "- Resident wardens provide regular medical and essential support to hostel residents.\n"
        "- The institution enforces a strict zero-tolerance no-ragging ethos.\n\n"
        "For personal, medical, or urgent safety support, contact the college administration directly (+91 9848823332 / info@srkrec.ac.in).\n\n"
        "Official Portal: https://www.srkrec.ac.in/"
    ),
    "faculty_directory": (
        "The academic leadership of SRKR Engineering College comprises the executive leadership, deans, and department heads:\n\n"
        "- **Executive Leadership:** Dr. M. Jagapathi Raju (Director), Dr. K.V. Murali Krishna Raju (Principal), Mr. Ch. Dileep Chakravarthy (Chief Administrative Officer).\n"
        "- **Deans:** Dr. V. Chandra Sekhar (Academics), Dr. T. Vamsi Nagaraju (R&D), Dr. K.R. Satyanarayana (Placements), Dr. S. Rajesh (IQAC), Dr. G. Harish Kumar Varma (Student Affairs & Alumni), Dr. B.S. Diwakar (Applied Sciences).\n"
        "- **Department Heads:** Dedicated HODs for Civil, CSE, ECE, EEE, Mechanical, IT, Management Studies, Mathematics, Chemistry, Physics, and English.\n\n"
        "Complete Leadership Directory: https://www.srkrec.ac.in/leadership/"
    ),
    "departments": (
        "SRKR Engineering College (Autonomous) comprises multiple academic and basic science departments:\n\n"
        "- **Engineering Departments:** Civil (civil@srkrec.ac.in), CSE (cse@srkrec.ac.in), ECE (ece@srkrec.ac.in), EEE (eee@srkrec.ac.in), Mechanical (mech@srkrec.ac.in), and IT (it@srkrec.ac.in).\n"
        "- **Specialized Programme Coordination:** CS & IT (csit@srkrec.ac.in), AI & ML (pc.aiml@srkrec.ac.in), AI & Data Science (hod.aids@srkrec.ac.in), CS & Business System (hod.csbs@srkrec.ac.in), CSE-IoT & Cyber Security (hod.cic@srkrec.ac.in), CS & Design (hod.csd@srkrec.ac.in).\n"
        "- **Management & Basic Sciences:** Management Studies (hod.bba@srkrec.ac.in), Mathematics (maths@srkrec.ac.in), Chemistry (chemistry@srkrec.ac.in), Physics (physics@srkrec.ac.in), and English (english@srkrec.ac.in).\n\n"
        "Complete Leadership Directory: https://www.srkrec.ac.in/leadership/"
    ),
    "fees": (
        "SRKR Engineering College publishes its official fee structure distinguishing tuition, university/examination charges, hostel, and transport charges.\n\n"
        "Because fee amounts vary by academic year, admission category, and programme, please refer directly to the official Fee Structure page:\n"
        "https://www.srkrec.ac.in/fee-structure/\n\n"
        "Note: The chatbot does not estimate or quote older fee amounts as current. For category-wise verification, contact the SRKR Admissions Office (+91 9848823332 / info@srkrec.ac.in)."
    ),
    "academic_calendar": (
        "SRKR Engineering College publishes regulation-specific academic calendars detailing the semester timeline:\n\n"
        "- Semester instructional periods and syllabus coverage\n"
        "- Continuous Internal Evaluation (Mid-term) schedules\n"
        "- Semester end practical and theory examination windows\n"
        "- Preparation holidays and semester breaks\n\n"
        "Official Academic Calendar: https://www.srkrec.ac.in/academic-calendar/"
    ),
    "events_and_annual_activities": (
        "SRKR Engineering College organizes marquee annual academic, technical, and sports events:\n\n"
        "- **SAMAGRA:** A 3-day flagship techno-cultural festival featuring national-level technical competitions in emerging areas, research conclaves, project expos, cultural events, and annual-day celebrations.\n"
        "- **JAITRA:** Annual national-level sports carnival conducted in January, attracting participants across universities for inter-collegiate tournaments and athletics.\n"
        "- **Departmental Hackathons:** Organized throughout the academic year focused on technology, sustainability, and domain challenges.\n\n"
        "Official Student Clubs & Events Page: https://www.srkrec.ac.in/student-clubs/"
    ),
    "model_papers": (
        "Model examination papers and previous years' resources are catalogued by regulation (e.g., R26, R23, R20) under the Autonomous Examinations Cell.\n\n"
        "Official Syllabus & Model Papers: https://www.srkrec.ac.in/syllabus/\n"
        "Examinations Portal: https://srkrexams.in"
    ),
    "syllabus_lookup_requirements": (
        "SRKR syllabus is structured by academic regulation (such as R26 for B.Tech first year, R23, R20) and engineering branch.\n\n"
        "To find the exact syllabus or subject list, please specify:\n"
        "1. Degree (B.Tech, M.Tech, BBA)\n"
        "2. Branch or Specialization (e.g., CSE, ECE, AI & ML, Mechanical)\n"
        "3. Regulation or Academic Year\n\n"
        "Official Syllabus Portal: https://www.srkrec.ac.in/syllabus/"
    )
}

# Clear out-of-scope triggers (unrelated domains)
OUT_OF_SCOPE_PATTERNS = [
    r'\b(flight|airline|flight\s*ticket|book\s*a\s*flight|train\s*ticket|irctc|redbus)\b',
    r'\b(weather|temperature|forecast|rain\s*today|rain\s*tomorrow)\b',
    r'\b(movie|cinema|actor|actress|bollywood|hollywood|box\s*office)\b',
    r'\b(crypto|bitcoin|ethereum|stock\s*market|share\s*market|sensex|nifty|nasdaq)\b',
    r'\b(pizza|burger|recipe|zomato|swiggy|cooking|food\s*recipe)\b',
    r'\b(hotel\s*booking|oyo|resort|vacation\s*package)\b',
]


class SRKRChatbotEngine:
    def __init__(
        self,
        model_name: str = "lstm",
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        similarity_threshold: float = config.SIMILARITY_THRESHOLD
    ):
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold
        self.current_model_name = ""
        self.current_model = None
        self.baseline_pipeline = None

        # Load Preprocessor and Knowledge Base
        self.preprocessor = SRKRDataPreprocessor(
            max_vocab_size=config.MAX_VOCAB_SIZE,
            max_len=config.MAX_SEQUENCE_LENGTH
        )
        self.preprocessor.load_artifacts(config.PROCESSED_DATA_DIR)
        self.knowledge_base = self.preprocessor.knowledge_base
        
        # Load independent knowledge base
        sep_kb_path = config.PROCESSED_DATA_DIR / "separate_knowledge_base.json"
        self.separate_kb = {}
        if sep_kb_path.exists():
            with open(sep_kb_path, "r", encoding="utf-8") as f:
                self.separate_kb = json.load(f)

        # Build TF-IDF Retrieval Index
        self._build_retrieval_index()

        # Load LSTM as the fixed DL model
        self.load_model(model_name)

    def _build_retrieval_index(self):
        """Construct TF-IDF index over all known intent patterns and KB items for semantic retrieval."""
        self.corpus_patterns = []
        self.corpus_tags = []

        with open(config.ORIGINAL_DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_intents = data.get("intents", []) if isinstance(data, dict) else data
        for item in raw_intents:
            tag = item.get("tag") or item.get("intent")
            canonical_tag = TAG_TO_KB_MAP.get(tag, tag)
            patterns = item.get("patterns") or item.get("questions") or []
            for p in patterns:
                p_str = str(p).strip()
                if p_str:
                    self.corpus_patterns.append(p_str)
                    self.corpus_tags.append(canonical_tag)

        raw_kb = data.get("knowledge_base", []) if isinstance(data, dict) else []
        for kb_item in raw_kb:
            kb_id = kb_item.get("id") or kb_item.get("tag")
            title = kb_item.get("title", "")
            aliases = kb_item.get("aliases", [])
            if title:
                self.corpus_patterns.append(title)
                self.corpus_tags.append(kb_id)
            for alias in aliases:
                if alias:
                    self.corpus_patterns.append(alias)
                    self.corpus_tags.append(kb_id)

        self.retrieval_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.retrieval_tfidf_matrix = self.retrieval_vectorizer.fit_transform(
            [clean_text(p) for p in self.corpus_patterns]
        )

    def load_model(self, model_name: str = "lstm"):
        """Load requested neural architecture (defaults to LSTM per Deep Learning Lab syllabus)."""
        model_key = model_name.lower().replace("-", "").replace(" ", "").replace("_", "")
        if "simplernn" in model_key or "rnn" in model_key:
            target = "simplernn"
            display_name = "SimpleRNN"
        elif "lstm" in model_key and "bi" not in model_key:
            target = "lstm"
            display_name = "LSTM"
        elif "gru" in model_key:
            target = "gru"
            display_name = "GRU"
        elif "baseline" in model_key or "tfidf" in model_key:
            target = "baseline_tfidf"
            display_name = "TF-IDF + Logistic Regression Baseline"
        else:
            target = "bilstm"
            display_name = "Bi-LSTM"

        if self.current_model_name == display_name and (self.current_model is not None or self.baseline_pipeline is not None):
            return

        if target == "baseline_tfidf":
            model_path = config.MODELS_DIR / "baseline_tfidf.joblib"
            if not model_path.exists():
                raise FileNotFoundError(f"Baseline model not found at {model_path}.")
            self.baseline_pipeline = joblib.load(model_path)
            self.current_model = None
        else:
            model_path = config.MODELS_DIR / f"{target}.keras"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}.")
            self.current_model = keras.models.load_model(str(model_path))
            self.baseline_pipeline = None

        self.current_model_name = display_name

    def retrieve_by_similarity(self, cleaned_query: str) -> Tuple[Optional[str], float, Optional[str]]:
        """Retrieve most similar known question using TF-IDF and cosine similarity with token overlap guard."""
        if not cleaned_query:
            return None, 0.0, None

        query_vec = self.retrieval_vectorizer.transform([cleaned_query])
        if query_vec.nnz == 0:
            return None, 0.0, None

        sims = cosine_similarity(query_vec, self.retrieval_tfidf_matrix).flatten()
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        best_tag = self.corpus_tags[best_idx]
        best_pattern = self.corpus_patterns[best_idx]

        # Require at least one non-trivial token overlap between cleaned_query and best_pattern
        q_tokens = set(clean_text(cleaned_query).split()) - {"what", "is", "the", "are", "of", "in", "to", "at", "for", "a", "an", "tell", "me", "about", "can", "you", "how", "do", "i"}
        p_tokens = set(clean_text(best_pattern).split())
        overlap = q_tokens.intersection(p_tokens)
        if not overlap and best_score < 0.60:
            return None, 0.0, None

        return best_tag, best_score, best_pattern

    def predict_intent(self, user_query: str) -> Tuple[str, float]:
        """Perform intent classification using loaded LSTM neural network."""
        cleaned = clean_text(user_query)
        if not cleaned:
            return "unknown", 0.0

        if self.baseline_pipeline is not None:
            vec = self.baseline_pipeline["vectorizer"].transform([cleaned])
            if vec.nnz == 0:
                return "unknown", 0.0
            probs = self.baseline_pipeline["classifier"].predict_proba(vec)[0]
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            pred_tag = self.preprocessor.idx2tag.get(pred_idx, "unknown")
            return pred_tag, confidence
        else:
            seq = self.preprocessor.texts_to_sequences([user_query])
            if all(idx in (0, 1) for idx in seq[0]):
                return "unknown", 0.0

            probs = self.current_model.predict(seq, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            pred_tag = self.preprocessor.idx2tag.get(pred_idx, "unknown")
            return pred_tag, confidence

    def format_kb_answer(self, raw_answer: str, source_urls: list = None, tag: str = None) -> str:
        """Present knowledge base information as a natural, helpful conversational answer."""
        if tag:
            canonical_tag = TAG_TO_KB_MAP.get(tag, tag)
            if canonical_tag in KB_REFINED_ANSWERS:
                return KB_REFINED_ANSWERS[canonical_tag]
            if tag in KB_REFINED_ANSWERS:
                return KB_REFINED_ANSWERS[tag]

        if not raw_answer:
            return config.DEFAULT_FALLBACK_RESPONSE

        text = raw_answer
        text = re.sub(r"The chatbot should.*?\.", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"Never invent a block-to-department mapping.*?\.", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"Treat them as reporting-period figures.*?\.", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"Confirm current accreditation.*before relying on time-sensitive claims\.", "", text).strip()
        text = text.strip()

        if source_urls and isinstance(source_urls, list) and len(source_urls) > 0:
            first_url = source_urls[0].strip()
            if first_url and first_url.startswith("http") and first_url not in text:
                text = f"{text}\n\nOfficial Source: {first_url}"

        return text

    def get_response(self, user_query: str) -> Dict[str, Any]:
        """
        Executes hybrid answer retrieval:
        1. Checks empty queries and out-of-scope domain guards.
        2. Checks short query keywords for immediate conversational answers.
        3. High-precision entity and domain-specific pattern routing.
        4. LSTM neural network intent prediction.
        5. TF-IDF semantic retrieval with token overlap validation.
        6. Canonical KB resolution and verified response formatting.
        """
        raw_query = user_query.strip()
        cleaned = clean_text(raw_query)

        # 1. Handle empty or blank input
        if not cleaned or len(cleaned.split()) == 0:
            return {
                "answer": "Please ask a question regarding SRKR Engineering College (e.g., admissions, courses, placements, hostels, library, facilities).",
                "intent": "empty_query",
                "confidence": 0.0,
                "model_used": self.current_model_name,
                "retrieval_score": 0.0,
                "status": "fallback",
                "category": "None"
            }

        # 2. Out-of-Scope Domain Guard (Unrelated questions)
        for pattern in OUT_OF_SCOPE_PATTERNS:
            if re.search(pattern, raw_query, re.I):
                return {
                    "answer": config.DEFAULT_FALLBACK_RESPONSE,
                    "intent": "out_of_scope",
                    "confidence": 1.0,
                    "model_used": self.current_model_name,
                    "retrieval_score": 0.0,
                    "matched_pattern": "out_of_scope",
                    "status": "out_of_scope_fallback",
                    "category": "Out of Scope"
                }

        # 3. Direct short query keywords
        query_lower = raw_query.lower().strip()
        norm_q = re.sub(r'[^a-zA-Z0-9\s]', '', query_lower).strip()

        matched_key = None
        if query_lower in SHORT_QUERY_RESPONSES:
            matched_key = query_lower
        elif norm_q in SHORT_QUERY_RESPONSES:
            matched_key = norm_q

        if matched_key:
            return {
                "answer": SHORT_QUERY_RESPONSES[matched_key],
                "intent": matched_key,
                "confidence": 1.0,
                "model_used": self.current_model_name,
                "retrieval_score": 1.0,
                "matched_pattern": matched_key,
                "status": "keyword_matched",
                "category": "Direct Inquiry"
            }

        # 4. Ambiguous single terms
        ambiguous_terms = {"help", "info", "information", "details", "college", "srkr"}
        if norm_q in ambiguous_terms:
            return {
                "answer": (
                    "SRKR Engineering College provides comprehensive information on admissions, courses, "
                    "departments, placements, hostels, library, campus transport, and facilities.\n\n"
                    "Could you please specify which topic or branch you would like to know about?"
                ),
                "intent": "clarification_needed",
                "confidence": 1.0,
                "model_used": self.current_model_name,
                "retrieval_score": 1.0,
                "matched_pattern": norm_q,
                "status": "clarification_requested",
                "category": "General Inquiry"
            }

        # 5. High-Precision Domain Pattern Routing for Specific Entities
        # Transport / Bus facilities
        if re.search(r'\b(bus|buses|bus\s+facility|bus\s+facilities|college\s+bus|transport|transportation|bus\s+timings|bus\s+route|bus\s+routes)\b', raw_query, re.I):
            return {
                "answer": SHORT_QUERY_RESPONSES["transport"],
                "intent": "transport",
                "confidence": 1.0,
                "model_used": self.current_model_name,
                "retrieval_score": 1.0,
                "matched_pattern": "transport",
                "status": "entity_routed",
                "category": "Transport"
            }

        # Leadership / Principal / Director
        if re.search(r'\b(who\s+is\s+(the\s+)?principal|principal\s+of\s+srkr|who\s+is\s+(the\s+)?director|director\s+of\s+srkr|executive\s+leadership|college\s+leadership|head\s+of\s+the\s+college|who\s+leads\s+srkr)\b', raw_query, re.I):
            return {
                "answer": KB_REFINED_ANSWERS["leadership_executive"],
                "intent": "leadership_executive",
                "confidence": 1.0,
                "model_used": self.current_model_name,
                "retrieval_score": 1.0,
                "matched_pattern": "leadership",
                "status": "entity_routed",
                "category": "Leadership"
            }

        # Scholarships & Fee reimbursement
        if re.search(r'\b(scholarship|scholarships|fee\s+reimbursement|financial\s+aid|jagananna)\b', raw_query, re.I):
            return {
                "answer": SHORT_QUERY_RESPONSES["scholarships"],
                "intent": "scholarships",
                "confidence": 1.0,
                "model_used": self.current_model_name,
                "retrieval_score": 1.0,
                "matched_pattern": "scholarships",
                "status": "entity_routed",
                "category": "Scholarships"
            }

        # Hostel Wardens & Contact
        if re.search(r'\b(warden|wardens|hostel\s+contact|hostel\s+phone|hostel\s+office|contact\s+hostel)\b', raw_query, re.I):
            return {
                "answer": KB_REFINED_ANSWERS["hostel_contacts"],
                "intent": "hostel_contacts",
                "confidence": 1.0,
                "model_used": self.current_model_name,
                "retrieval_score": 1.0,
                "matched_pattern": "hostel_contacts",
                "status": "entity_routed",
                "category": "Hostel"
            }

        # Placements statistics & Top package
        if re.search(r'\b(placement\s+percentage|placement\s+stats|placement\s+statistics|top\s+package|highest\s+package|average\s+package|recruiters|companies\s+visited)\b', raw_query, re.I):
            return {
                "answer": KB_REFINED_ANSWERS["placement_statistics"],
                "intent": "placement_statistics",
                "confidence": 1.0,
                "model_used": self.current_model_name,
                "retrieval_score": 1.0,
                "matched_pattern": "placement_statistics",
                "status": "entity_routed",
                "category": "Placements"
            }

        # SAMAGRA fest
        if re.search(r'\b(samagra|techno\s*cultural\s*fest|annual\s*fest)\b', raw_query, re.I):
            return {
                "answer": KB_REFINED_ANSWERS["samagra_festival"],
                "intent": "samagra_festival",
                "confidence": 1.0,
                "model_used": self.current_model_name,
                "retrieval_score": 1.0,
                "matched_pattern": "samagra",
                "status": "entity_routed",
                "category": "Campus Events"
            }

        # JAITRA sports carnival
        if re.search(r'\b(jaitra|sports\s*carnival|sports\s*fest|annual\s*sports\s*meet)\b', raw_query, re.I):
            return {
                "answer": KB_REFINED_ANSWERS["jaitra_sports_carnival"],
                "intent": "jaitra_sports_carnival",
                "confidence": 1.0,
                "model_used": self.current_model_name,
                "retrieval_score": 1.0,
                "matched_pattern": "jaitra_sports_carnival",
                "status": "entity_routed",
                "category": "Sports"
            }

        # Hackathons
        if re.search(r'\b(hackathon|hackathons|project\s+expo|coding\s+contest)\b', raw_query, re.I):
            return {
                "answer": KB_REFINED_ANSWERS["hackathons_project_expos"],
                "intent": "hackathons_project_expos",
                "confidence": 1.0,
                "model_used": self.current_model_name,
                "retrieval_score": 1.0,
                "matched_pattern": "hackathons",
                "status": "entity_routed",
                "category": "Events"
            }

        # Technical Clubs
        if re.search(r'\b(technical\s+club|technical\s+clubs|ieee|acm|csi|coding\s+club|robotics\s+club|drone\s+club)\b', raw_query, re.I):
            return {
                "answer": KB_REFINED_ANSWERS["student_clubs_technical"],
                "intent": "student_clubs_technical",
                "confidence": 1.0,
                "model_used": self.current_model_name,
                "retrieval_score": 1.0,
                "matched_pattern": "technical_clubs",
                "status": "entity_routed",
                "category": "Clubs"
            }

        # ECE Labs
        if re.search(r'\b(ece\s+lab|ece\s+laboratories|electronics\s+lab|vlsi\s+lab|dsp\s+lab)\b', raw_query, re.I):
            return {
                "answer": KB_REFINED_ANSWERS["ece_labs"],
                "intent": "ece_labs",
                "confidence": 1.0,
                "model_used": self.current_model_name,
                "retrieval_score": 1.0,
                "matched_pattern": "ece_labs",
                "status": "entity_routed",
                "category": "Laboratories"
            }

        # AIML / AI & ML programmes
        if re.search(r'\b(aiml|ai\s*&\s*ml|artificial\s+intelligence)\b', raw_query, re.I):
            return {
                "answer": SHORT_QUERY_RESPONSES["aiml"],
                "intent": "aiml_programme",
                "confidence": 1.0,
                "model_used": self.current_model_name,
                "retrieval_score": 1.0,
                "matched_pattern": "aiml",
                "status": "entity_routed",
                "category": "Programmes"
            }

        # Courses / Branches offered
        if re.search(r'\b(what\s+courses|which\s+(engineering\s+)?branches|what\s+branches|tell\s+me\s+about\s+(the\s+)?programmes|available\s+courses|courses\s+offered|courses\s+available|programmes\s+offered)\b', raw_query, re.I):
            return {
                "answer": SHORT_QUERY_RESPONSES["courses"],
                "intent": "courses_overview",
                "confidence": 1.0,
                "model_used": self.current_model_name,
                "retrieval_score": 1.0,
                "matched_pattern": "courses",
                "status": "entity_routed",
                "category": "Programmes"
            }

        # Hostel general inquiries
        if re.search(r'\b(hostel\s+facilities|hostel\s+accommodation|hostels\s+available|hostel\s+admission)\b', raw_query, re.I):
            return {
                "answer": SHORT_QUERY_RESPONSES["hostel"],
                "intent": "hostel_overview",
                "confidence": 1.0,
                "model_used": self.current_model_name,
                "retrieval_score": 1.0,
                "matched_pattern": "hostel",
                "status": "entity_routed",
                "category": "Hostel"
            }

        # Library facilities
        if re.search(r'\b(central\s+library|facilities\s+in\s+(the\s+)?library|library\s+facilities|about\s+the\s+library|library\s+books)\b', raw_query, re.I):
            return {
                "answer": SHORT_QUERY_RESPONSES["library"],
                "intent": "library",
                "confidence": 1.0,
                "model_used": self.current_model_name,
                "retrieval_score": 1.0,
                "matched_pattern": "library",
                "status": "entity_routed",
                "category": "Library"
            }

        # Anti-ragging
        if re.search(r'\b(anti\s*ragging|ragging\s*policy|is\s*ragging\s*allowed|ragging\s*rules)\b', raw_query, re.I):
            return {
                "answer": KB_REFINED_ANSWERS["ragging"],
                "intent": "ragging",
                "confidence": 1.0,
                "model_used": self.current_model_name,
                "retrieval_score": 1.0,
                "matched_pattern": "ragging",
                "status": "entity_routed",
                "category": "Campus Rules"
            }

        # Examination results / Portal
        if re.search(r'\b(semester\s+results|view\s+results|exam\s+results|srkrexams|revaluation|grade\s+cards|sgpa|cgpa)\b', raw_query, re.I):
            return {
                "answer": KB_REFINED_ANSWERS["examinations"],
                "intent": "examinations",
                "confidence": 1.0,
                "model_used": self.current_model_name,
                "retrieval_score": 1.0,
                "matched_pattern": "examinations",
                "status": "entity_routed",
                "category": "Examinations"
            }

        # College location / identity
        if re.search(r'\b(where\s+is\s+the\s+college\s+located|where\s+is\s+srkr|college\s+location|campus\s+address|campus\s+size|how\s+many\s+acres)\b', raw_query, re.I):
            return {
                "answer": KB_REFINED_ANSWERS["college_identity"],
                "intent": "college_identity",
                "confidence": 1.0,
                "model_used": self.current_model_name,
                "retrieval_score": 1.0,
                "matched_pattern": "college_identity",
                "status": "entity_routed",
                "category": "About SRKR"
            }

        # 6. LSTM Deep Learning Prediction
        pred_tag, confidence = self.predict_intent(user_query)

        # 7. TF-IDF Similarity Retrieval over verified knowledge base
        retrieved_tag, sim_score, matched_pattern = self.retrieve_by_similarity(cleaned)

        # 8. Hybrid Decision Gate
        chosen_tag = None
        status = "matched"

        if confidence >= self.confidence_threshold:
            if pred_tag == retrieved_tag or confidence >= 0.65:
                chosen_tag = pred_tag
                status = "deep_learning_matched"
            elif sim_score >= self.similarity_threshold:
                chosen_tag = retrieved_tag if sim_score > 0.50 else pred_tag
                status = "hybrid_aligned"
            else:
                chosen_tag = pred_tag
                status = "deep_learning_matched"
        else:
            if sim_score >= self.similarity_threshold:
                chosen_tag = retrieved_tag
                status = "retrieval_fallback_matched"
            else:
                chosen_tag = None
                status = "fallback"

        # 9. Fetch verified answer from knowledge base using canonical mapping
        answer = None
        category = "General"

        if chosen_tag:
            canonical_tag = TAG_TO_KB_MAP.get(chosen_tag, chosen_tag)

            if canonical_tag in KB_REFINED_ANSWERS:
                answer = KB_REFINED_ANSWERS[canonical_tag]
                category = "Campus Knowledge"
            elif canonical_tag in self.separate_kb:
                kb_entry = self.separate_kb[canonical_tag]
                raw_ans = kb_entry.get("answer", "")
                urls = kb_entry.get("source_urls", [])
                answer = self.format_kb_answer(raw_ans, urls, tag=canonical_tag)
                category = kb_entry.get("category", "Campus Knowledge")
            elif chosen_tag in self.separate_kb:
                kb_entry = self.separate_kb[chosen_tag]
                raw_ans = kb_entry.get("answer", "")
                urls = kb_entry.get("source_urls", [])
                answer = self.format_kb_answer(raw_ans, urls, tag=chosen_tag)
                category = kb_entry.get("category", "Campus Knowledge")
            elif chosen_tag in self.knowledge_base:
                kb_entry = self.knowledge_base[chosen_tag]
                responses = kb_entry.get("responses", [])
                raw_ans = responses[0] if responses else ""
                urls = []
                if canonical_tag in self.separate_kb:
                    urls = self.separate_kb[canonical_tag].get("source_urls", [])
                answer = self.format_kb_answer(raw_ans, urls, tag=chosen_tag)
                category = kb_entry.get("category", "General")

        if not answer:
            answer = config.DEFAULT_FALLBACK_RESPONSE
            category = "Out of Scope / Unknown"

        return {
            "answer": answer,
            "intent": chosen_tag if chosen_tag else pred_tag,
            "confidence": round(float(confidence), 4),
            "model_used": self.current_model_name,
            "retrieval_score": round(float(sim_score), 4),
            "matched_pattern": matched_pattern,
            "status": status,
            "category": category
        }
