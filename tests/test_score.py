import unittest
import json
import sys
import os
from unittest.mock import patch, MagicMock
import time

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient
from app.main import app

class TestResumeScorer(unittest.TestCase):
    
    def setUp(self):
        """Set up test client"""
        self.client = TestClient(app)
        
        # Test data
        self.high_score_resume = {
            "student_id": "test_001",
            "goal": "Amazon SDE",
            "resume_text": """
            Software Development Engineer with 3 years of experience in Java, Python, and JavaScript.
            Strong expertise in data structures and algorithms, with hands-on experience in system design.
            Proficient in SQL database management and REST API development.
            Experience with Git version control, AWS cloud services, and microservices architecture.
            Built scalable web applications using Spring Boot and React.
            Strong problem-solving skills with experience in debugging and testing.
            """
        }
        
        self.low_score_resume = {
            "student_id": "test_002", 
            "goal": "Amazon SDE",
            "resume_text": """
            Graphic Designer with 5 years of experience in visual design and branding.
            Proficient in Adobe Photoshop, Illustrator, and InDesign.
            Experience in creating marketing materials, logos, and website layouts.
            Strong understanding of color theory and typography.
            Worked with clients in various industries including fashion and hospitality.
            """
        }
        
        self.ml_resume = {
            "student_id": "test_003",
            "goal": "ML Internship", 
            "resume_text": """
            Data Science student with strong background in Python and machine learning.
            Experience with NumPy, Pandas, Scikit-learn, and TensorFlow.
            Completed projects in computer vision and natural language processing.
            Strong foundation in statistics and linear algebra.
            Published research in deep learning and neural networks.
            """
        }
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
    
    def test_version_endpoint(self):
        """Test version endpoint"""
        response = self.client.get("/version")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("version", data)
        self.assertIn("supported_goals", data)
        self.assertIn("minimum_score_threshold", data)
        self.assertIsInstance(data["supported_goals"], list)
        self.assertGreater(len(data["supported_goals"]), 0)
    
    def test_score_high_match_resume(self):
        """Test scoring a resume that should match well"""
        response = self.client.post("/score", json=self.high_score_resume)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        
        # Validate response structure
        self.assertIn("score", data)
        self.assertIn("matched_skills", data)
        self.assertIn("missing_skills", data)
        self.assertIn("suggested_learning_path", data)
        
        # Validate data types
        self.assertIsInstance(data["score"], float)
        self.assertIsInstance(data["matched_skills"], list)
        self.assertIsInstance(data["missing_skills"], list)
        self.assertIsInstance(data["suggested_learning_path"], list)
        
        # Validate score range
        self.assertGreaterEqual(data["score"], 0.0)
        self.assertLessEqual(data["score"], 1.0)
        
        # High match resume should have good score
        self.assertGreater(data["score"], 0.5)
        
        # Should have some matched skills
        self.assertGreater(len(data["matched_skills"]), 0)
        
        print(f"High score resume result: {data}")
    
    def test_score_low_match_resume(self):
        """Test scoring a resume that should not match well"""
        response = self.client.post("/score", json=self.low_score_resume)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        
        # Validate response structure
        self.assertIn("score", data)
        self.assertIn("matched_skills", data)
        self.assertIn("missing_skills", data)
        self.assertIn("suggested_learning_path", data)
        
        # Low match resume should have lower score
        self.assertLess(data["score"], 0.7)
        
        # Should have missing skills
        self.assertGreater(len(data["missing_skills"]), 0)
        
        # Should have learning path suggestions
        self.assertGreater(len(data["suggested_learning_path"]), 0)
        
        print(f"Low score resume result: {data}")
    
    def test_ml_internship_goal(self):
        """Test scoring for ML Internship goal"""
        response = self.client.post("/score", json=self.ml_resume)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        
        # Should have good score for relevant ML resume
        self.assertGreater(data["score"], 0.6)
        
        # Should match relevant ML skills
        ml_skills = ["Python", "Machine Learning", "NumPy", "Pandas", "TensorFlow"]
        matched_count = sum(1 for skill in ml_skills if skill in data["matched_skills"])
        self.assertGreater(matched_count, 2)
        
        print(f"ML internship resume result: {data}")
    
    def test_response_time(self):
        """Test that response time is under 1.5 seconds"""
        start_time = time.time()
        response = self.client.post("/score", json=self.high_score_resume)
        end_time = time.time()
        
        response_time = end_time - start_time
        self.assertLess(response_time, 1.5)
        self.assertEqual(response.status_code, 200)
        
        print(f"Response time: {response_time:.3f} seconds")
    
    def test_invalid_goal(self):
        """Test with unsupported goal"""
        invalid_request = {
            "student_id": "test_004",
            "goal": "Invalid Goal",
            "resume_text": "Some resume text"
        }
        
        response = self.client.post("/score", json=invalid_request)
        self.assertEqual(response.status_code, 400)
        
        data = response.json()
        self.assertIn("detail", data)
    
    def test_empty_resume(self):
        """Test with empty resume text"""
        empty_request = {
            "student_id": "test_005",
            "goal": "Amazon SDE",
            "resume_text": ""
        }
        
        response = self.client.post("/score", json=empty_request)
        self.assertEqual(response.status_code, 422)  # Validation error
    
    def test_missing_fields(self):
        """Test with missing required fields"""
        incomplete_request = {
            "student_id": "test_006",
            "goal": "Amazon SDE"
            # Missing resume_text
        }
        
        response = self.client.post("/score", json=incomplete_request)
        self.assertEqual(response.status_code, 422)  # Validation error
    
    def test_malformed_json(self):
        """Test with malformed JSON"""
        response = self.client.post(
            "/score",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        self.assertEqual(response.status_code, 422)
    
    def test_frontend_endpoint(self):
        """Test that frontend endpoint returns HTML"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
    
    def test_multiple_goals(self):
        """Test scoring the same resume against different goals"""
        base_request = {
            "student_id": "test_007",
            "resume_text": """
            Python developer with experience in machine learning and web development.
            Skilled in NumPy, Pandas, Django, and Flask.
            Experience with data analysis and REST API development.
            """
        }
        
        goals = ["Amazon SDE", "ML Internship"]
        results = {}
        
        for goal in goals:
            request_data = {**base_request, "goal": goal}
            response = self.client.post("/score", json=request_data)
            self.assertEqual(response.status_code, 200)
            results[goal] = response.json()
        
        # Results should be different for different goals
        self.assertNotEqual(results["Amazon SDE"]["score"], results["ML Internship"]["score"])
        
        print(f"Multi-goal results: {results}")

def run_performance_test():
    """Run performance test with multiple concurrent requests"""
    import threading
    import time
    
    client = TestClient(app)
    results = []
    
    def make_request():
        start_time = time.time()
        response = client.post("/score", json={
            "student_id": "perf_test",
            "goal": "Amazon SDE",
            "resume_text": "Java Python SQL REST API Git"
        })
        end_time = time.time()
        results.append({
            "status_code": response.status_code,
            "response_time": end_time - start_time
        })
    
    # Create 10 concurrent requests
    threads = []
    for i in range(10):
        thread = threading.Thread(target=make_request)
        threads.append(thread)
    
    start_time = time.time()
    for thread in threads:
        thread.start()
    
    for thread in threads:
        thread.join()
    
    end_time = time.time()
    
    print(f"\nPerformance Test Results:")
    print(f"Total time for 10 requests: {end_time - start_time:.3f} seconds")
    print(f"Average response time: {sum(r['response_time'] for r in results) / len(results):.3f} seconds")
    print(f"All requests successful: {all(r['status_code'] == 200 for r in results)}")

if __name__ == "__main__":
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    # Run performance test
    run_performance_test()