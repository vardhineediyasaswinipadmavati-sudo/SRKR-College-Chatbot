import unittest
import numpy as np
from pathlib import Path
import config
from data_preprocessing import SRKRDataPreprocessor, clean_text, tokenize
from chatbot import SRKRChatbotEngine

class TestSRKRChatbot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # LSTM model loaded automatically by default (per DL Lab syllabus)
        cls.engine = SRKRChatbotEngine()

    def test_lstm_loaded_by_default(self):
        """Verify LSTM is the default fixed model loaded."""
        self.assertEqual(self.engine.current_model_name, "LSTM")
        self.assertIsNotNone(self.engine.current_model)

    def test_preprocessor_artifacts_exist(self):
        """Verify preprocessor artifacts and knowledge bases exist."""
        self.assertTrue((config.PROCESSED_DATA_DIR / "preprocessor_config.json").exists())
        self.assertTrue((config.PROCESSED_DATA_DIR / "knowledge_base.json").exists())
        self.assertTrue((config.PROCESSED_DATA_DIR / "separate_knowledge_base.json").exists())
        self.assertTrue((config.PROCESSED_DATA_DIR / "dataset_stats.json").exists())

    def test_short_query_map(self):
        """Test short query 'map' produces exact verified answer with map link."""
        res = self.engine.get_response("map")
        answer = res["answer"]
        self.assertIn("N Block", answer)
        self.assertIn("Silver Jubilee Block", answer)
        self.assertIn("Civil Department", answer)
        self.assertIn("WET Centre", answer)
        self.assertIn("https://mapcarta.com/W851081636", answer)
        # Verify no debug keywords in answer
        for debug_term in ["Intent:", "Confidence:", "Retrieval Score:", "Category:", "Status:", "retrieval_fallback_matched"]:
            self.assertNotIn(debug_term, answer)

    def test_short_query_hostel(self):
        """Test short query 'hostel'."""
        res = self.engine.get_response("hostel")
        answer = res["answer"]
        self.assertIn("hostel", answer.lower())
        self.assertIn("boys and girls", answer.lower())
        self.assertIn("8074874235", answer)
        self.assertIn("7995383711", answer)
        self.assertIn("hostels@srkrec.ac.in", answer)
        self.assertIn("https://www.srkrec.ac.in/hostels/", answer)

    def test_short_query_library(self):
        """Test short query 'library'."""
        res = self.engine.get_response("library")
        answer = res["answer"]
        self.assertIn("library", answer.lower())
        self.assertIn("rfid", answer.lower())
        self.assertIn("4,160", answer)
        self.assertIn("https://www.srkrec.ac.in/library/", answer)

    def test_short_query_placements(self):
        """Test short query 'placements'."""
        res = self.engine.get_response("placements")
        answer = res["answer"]
        self.assertIn("placement", answer.lower())
        self.assertIn("87.3%", answer)
        self.assertIn("19.00", answer)
        self.assertIn("https://www.srkrec.ac.in/placements-team/", answer)

    def test_short_query_aiml(self):
        """Test short query 'AIML'."""
        res = self.engine.get_response("AIML")
        answer = res["answer"]
        self.assertIn("artificial intelligence", answer.lower())
        self.assertIn("https://www.srkrec.ac.in/programmes/", answer)

    def test_short_query_departments(self):
        """Test short query 'departments'."""
        res = self.engine.get_response("departments")
        answer = res["answer"]
        self.assertIn("civil", answer.lower())
        self.assertIn("cse", answer.lower())
        self.assertIn("https://www.srkrec.ac.in/leadership/", answer)

    def test_short_query_scholarships(self):
        """Test short query 'scholarships' correctly explains limitation without inventing schemes."""
        res = self.engine.get_response("scholarships")
        answer = res["answer"]
        self.assertIn("not currently catalogued", answer.lower())
        self.assertIn("https://www.srkrec.ac.in/fee-structure/", answer)

    def test_phrasings_courses(self):
        """Verify different phrasings of course questions return comprehensive programme information."""
        phrasings = [
            "What courses are available at SRKR?",
            "Which engineering branches does SRKR offer?",
            "Tell me about the programmes offered by SRKR."
        ]
        for query in phrasings:
            res = self.engine.get_response(query)
            ans = res["answer"].lower()
            self.assertTrue("b.tech" in ans or "programmes" in ans or "computer science" in ans or "engineering" in ans,
                            f"Query '{query}' did not return course info: {res['answer']}")

    def test_suggested_questions(self):
        """Verify all suggested questions from UI return relevant responses."""
        suggestions = [
            "What courses are offered at SRKR?",
            "Tell me about the AIML department.",
            "What hostel facilities are available?",
            "What are the placement opportunities?",
            "What facilities are available in the library?",
            "How can I contact the college?"
        ]
        for q in suggestions:
            res = self.engine.get_response(q)
            self.assertTrue(len(res["answer"]) > 30, f"Answer too short for '{q}'")
            # Verify no debug keywords leak
            for debug_term in ["Intent:", "Confidence:", "Retrieval Score:", "Category:", "Status:", "retrieval_fallback_matched"]:
                self.assertNotIn(debug_term, res["answer"])

    def test_longer_question_admissions(self):
        """Test longer question regarding B.Tech admissions."""
        query = "How can a student get admission into B.Tech at SRKR College?"
        res = self.engine.get_response(query)
        answer = res["answer"]
        self.assertTrue(len(answer) > 30)
        self.assertTrue("admission" in answer.lower() or "eapcet" in answer.lower() or "srkr" in answer.lower())

    def test_longer_question_courses(self):
        """Test longer question regarding available courses."""
        query = "Which engineering degrees and branches are available at SRKR?"
        res = self.engine.get_response(query)
        answer = res["answer"]
        self.assertTrue("b.tech" in answer.lower() or "programmes" in answer.lower() or "computer science" in answer.lower())

    def test_out_of_scope_query_fallback(self):
        """Test completely unknown out-of-scope question receives polite fallback."""
        query = "What is the stock price of Apple on NASDAQ today?"
        res = self.engine.get_response(query)
        answer = res["answer"]
        self.assertIn("verified information", answer.lower())
        self.assertIn("https://www.srkrec.ac.in", answer)

    def test_no_debug_info_in_responses(self):
        """Verify no internal debug tags leak into conversational responses."""
        test_queries = ["map", "hostel", "fees", "placements", "AIML", "courses", "Where is the library?", "departments", "scholarships"]
        forbidden_tags = [
            "Intent:", "Confidence:", "Retrieval Score:", "Category:", "Status:",
            "retrieval_fallback_matched", "deep_learning_matched", "hybrid_aligned"
        ]
        for q in test_queries:
            ans = self.engine.get_response(q)["answer"]
            for tag in forbidden_tags:
                self.assertNotIn(tag, ans, f"Forbidden debug tag '{tag}' found in answer for query '{q}'")

    def test_all_models_preserved(self):
        """Verify all other models (SimpleRNN, GRU, Bi-LSTM, Baseline) are preserved and loadable for evaluation."""
        for m in ["simplernn", "gru", "bilstm", "baseline"]:
            self.engine.load_model(m)
            self.assertIsNotNone(self.engine.current_model if m != "baseline" else self.engine.baseline_pipeline)
        # Restore fixed LSTM
        self.engine.load_model("lstm")
        self.assertEqual(self.engine.current_model_name, "LSTM")

    def test_comprehensive_25_queries(self):
        """Verify 25 representative queries across all major college categories."""
        test_cases = [
            ("What courses are available at SRKR?", ["b.tech", "programmes", "srkrec.ac.in"]),
            ("Which engineering branches does SRKR offer?", ["cse", "ece", "mechanical"]),
            ("Tell me about the programmes offered by SRKR.", ["undergraduate", "postgraduate"]),
            ("Tell me about AIML department", ["artificial intelligence", "aiml"]),
            ("How do I get admission into SRKR?", ["eapcet", "admission", "srkrec.ac.in"]),
            ("What is the placement percentage?", ["87.3%", "19.00", "placements"]),
            ("What hostel facilities are provided?", ["hostel", "boys and girls", "warden"]),
            ("What bus facilities are available?", ["bus", "bhimavaram", "transport"]),
            ("Who is the Principal of SRKR?", ["principal", "dr. k.v. murali krishna raju"]),
            ("Tell me about the central library", ["library", "4,160", "rfid"]),
            ("What are the lab facilities in ECE?", ["ece", "laboratories", "electronic devices"]),
            ("Are there any scholarships available?", ["not currently catalogued", "fee-structure"]),
            ("What is the college fee structure?", ["fee structure", "tuition", "srkrec.ac.in"]),
            ("Tell me about SAMAGRA fest", ["samagra", "techno-cultural", "festival"]),
            ("Tell me about JAITRA sports carnival", ["jaitra", "sports carnival", "january"]),
            ("How do I view my semester results?", ["examinations", "srkrexams.in"]),
            ("What is the syllabus for R23 B.Tech?", ["r23", "syllabus", "srkrec.ac.in"]),
            ("What is the contact number for hostel warden?", ["8074874235", "9492123311", "7995383711"]),
            ("Where is the college located?", ["bhimavaram", "china amiram", "30 acres"]),
            ("What technical clubs can I join?", ["ieee", "acm", "clubs"]),
            ("Can you book me a flight ticket to Delhi?", ["verified information", "info@srkrec.ac.in"]),
            ("What is the weather in Bhimavaram today?", ["verified information", "info@srkrec.ac.in"]),
            ("Tell me about Mechanical Engineering department", ["mechanical", "1980", "cad/cam"]),
            ("What is the top placement package?", ["19.00 lpa", "at&t", "87.3%"]),
            ("What are the anti-ragging measures at SRKR?", ["anti-ragging", "no-ragging", "hostel"])
        ]
        for query, expected_keywords in test_cases:
            res = self.engine.get_response(query)
            ans = res["answer"].lower()
            for kw in expected_keywords:
                self.assertIn(kw.lower(), ans, f"Keyword '{kw}' missing for query '{query}'. Answer: {ans}")
            # Ensure no debug leakage
            for debug_term in ["Intent:", "Confidence:", "Retrieval Score:", "Category:", "Status:", "retrieval_fallback_matched"]:
                self.assertNotIn(debug_term, res["answer"])

if __name__ == "__main__":
    unittest.main()
