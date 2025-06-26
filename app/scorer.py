import json
import joblib
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ResumeScorer:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.vectorizers = {}
        self.goals_skills = {}
        
        # Load goals and skills
        self._load_goals()
        
        # Load or train models
        self._load_models()
    
    def _load_goals(self):
        """Load goals and required skills from goals.json"""
        try:
            with open('data/goals.json', 'r') as f:
                self.goals_skills = json.load(f)
            logger.info("Goals and skills loaded successfully")
        except FileNotFoundError:
            logger.error("goals.json not found")
            # Create default goals if file doesn't exist
            self.goals_skills = {
                "Amazon SDE": ["Java", "Python", "Data Structures", "Algorithms", "System Design", "SQL", "REST APIs", "Git"],
                "ML Internship": ["Python", "Machine Learning", "Deep Learning", "NumPy", "Pandas", "Scikit-learn", "TensorFlow", "Statistics"],
                "GATE ECE": ["Digital Electronics", "Signal Processing", "Communication Systems", "Control Systems", "Electromagnetics", "Network Theory"]
            }
            # Save default goals
            os.makedirs('data', exist_ok=True)
            with open('data/goals.json', 'w') as f:
                json.dump(self.goals_skills, f, indent=2)
    
    def _load_models(self):
        """Load trained models or create new ones if they don't exist"""
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
        os.makedirs(model_dir, exist_ok=True)
        
        for goal in self.goals_skills.keys():
            model_file = os.path.join(model_dir, f"{goal.lower().replace(' ', '_')}_model.pkl")
            vectorizer_file = os.path.join(model_dir, f"{goal.lower().replace(' ', '_')}_vectorizer.pkl")
            
            if os.path.exists(model_file) and os.path.exists(vectorizer_file):
                logger.info(f"Loading model from {model_file}")
                logger.info(f"Loading vectorizer from {vectorizer_file}")
                try:
                    self.models[goal] = joblib.load(model_file)
                    self.vectorizers[goal] = joblib.load(vectorizer_file)
                    logger.info(f"Loaded model for {goal}")
                except Exception as e:
                    logger.warning(f"Failed to load model for {goal}: {str(e)}")
                    self._train_model(goal)
            else:
                logger.info(f"Training new model for {goal}")
                self._train_model(goal)
    
    def _generate_training_data(self, goal):
        """Generate synthetic training data for a goal"""
        positive_samples = []
        negative_samples = []
        
        # Get skills for the goal
        goal_skills = self.goals_skills.get(goal, [])
        
        if goal == "Amazon SDE":
            positive_samples = [
                "Software engineer with 3 years experience in Java, Python, and JavaScript. Skilled in data structures, algorithms, system design, and SQL. Experience with REST APIs, microservices, Git, and AWS.",
                "Full-stack developer proficient in Java, Spring Boot, React, and MySQL. Strong foundation in data structures and algorithms. Experience with distributed systems and cloud technologies.",
                "Computer science graduate with expertise in Python, Java, C++. Solid understanding of algorithms, data structures, system design, and database management. Internship experience at tech companies.",
                "Backend developer with experience in Java, Node.js, and databases. Knowledge of system design, REST APIs, and version control with Git. Strong problem-solving skills.",
                "Software development intern with hands-on experience in Python, Java, and web technologies. Understanding of algorithms, data structures, and software engineering principles."
            ]
            
            negative_samples = [
                "Mechanical engineer with expertise in CAD design, AutoCAD, SolidWorks. Experience in manufacturing processes and quality control. No programming background.",
                "Marketing professional with skills in social media marketing, content creation, and brand management. Experience with Google Analytics and digital advertising.",
                "Civil engineer specializing in structural design and project management. Proficient in structural analysis software and construction management.",
                "Graphic designer with expertise in Photoshop, Illustrator, and UI/UX design. Experience in branding and visual communication.",
                "Financial analyst with strong background in Excel, financial modeling, and investment analysis. CFA certification and banking experience."
            ]
        
        elif goal == "ML Internship":
            positive_samples = [
                "Data science student with experience in Python, machine learning, and deep learning. Proficient in NumPy, Pandas, Scikit-learn, and TensorFlow. Strong mathematical background.",
                "Computer science graduate with focus on artificial intelligence. Experience with Python, R, machine learning algorithms, and statistical analysis. Projects in computer vision and NLP.",
                "Research assistant with background in machine learning and data analysis. Skilled in Python, PyTorch, Keras, and data visualization. Published research in ML conferences.",
                "Data analyst with expertise in Python, machine learning, and statistical modeling. Experience with Scikit-learn, XGBoost, and feature engineering techniques.",
                "AI enthusiast with hands-on experience in deep learning, neural networks, and computer vision. Proficient in Python, TensorFlow, and OpenCV."
            ]
            
            negative_samples = [
                "Web developer with expertise in HTML, CSS, JavaScript, and React. Full-stack development experience with Node.js and MongoDB.",
                "Network administrator with skills in Cisco networking, server management, and cybersecurity. CompTIA and Cisco certifications.",
                "Mobile app developer specializing in Android and iOS development. Experience with Java, Swift, and React Native.",
                "Database administrator with expertise in SQL Server, Oracle, and database optimization. Experience in data warehousing and ETL processes.",
                "DevOps engineer with experience in Docker, Kubernetes, Jenkins, and cloud platforms. Infrastructure automation and CI/CD pipelines."
            ]
        
        elif goal == "GATE ECE":
            positive_samples = [
                "Electronics engineering student with strong foundation in digital electronics, signal processing, and communication systems. Experience with MATLAB and circuit design.",
                "ECE graduate with expertise in analog and digital circuits, microprocessors, and control systems. Strong mathematical background and laboratory experience.",
                "Electronics engineer with knowledge of VLSI design, embedded systems, and signal processing. Experience with Verilog, MATLAB, and PCB design.",
                "Communication systems engineer with background in wireless communication, antenna design, and RF circuits. Strong theoretical and practical knowledge.",
                "ECE student with focus on power electronics, control systems, and electrical machines. Experience with simulation tools and circuit analysis."
            ]
            
            negative_samples = [
                "Software developer with expertise in Java, Python, and web development. Full-stack development experience with modern frameworks.",
                "Mechanical engineer with skills in CAD design, thermodynamics, and manufacturing processes. Experience with AutoCAD and SolidWorks.",
                "Computer science graduate with background in algorithms, data structures, and software engineering. Programming experience in multiple languages.",
                "Data scientist with expertise in machine learning, statistics, and data analysis. Proficient in Python, R, and data visualization tools.",
                "Civil engineer specializing in structural design and construction management. Experience with structural analysis software and project planning."
            ]
        
        # Create training data format
        training_data = []
        
        for sample in positive_samples:
            training_data.append({
                "goal": goal,
                "resume_text": sample,
                "label": 1
            })
        
        for sample in negative_samples:
            training_data.append({
                "goal": goal,
                "resume_text": sample,
                "label": 0
            })
        
        # Add some additional synthetic samples by mixing skills
        import random
        for _ in range(20):
            # Create positive samples with goal skills
            skills_subset = random.sample(goal_skills, min(len(goal_skills), random.randint(3, 6)))
            positive_text = f"Professional with experience in {', '.join(skills_subset)}. Strong background and practical knowledge."
            training_data.append({
                "goal": goal,
                "resume_text": positive_text,
                "label": 1
            })
            
            # Create negative samples with random skills
            all_skills = []
            for g_skills in self.goals_skills.values():
                all_skills.extend(g_skills)
            other_skills = [s for s in all_skills if s not in goal_skills]
            if other_skills:
                random_skills = random.sample(other_skills, min(len(other_skills), random.randint(2, 4)))
                negative_text = f"Experienced in {', '.join(random_skills)}. Professional background in various domains."
                training_data.append({
                    "goal": goal,
                    "resume_text": negative_text,
                    "label": 0
                })
        
        return training_data
    
    def _train_model(self, goal):
        """Train a model for a specific goal"""
        try:
            # Generate training data
            training_data = self._generate_training_data(goal)
            
            # Prepare data
            texts = [item['resume_text'] for item in training_data]
            labels = [item['label'] for item in training_data]
            
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                lowercase=True
            )
            
            # Vectorize texts
            X = vectorizer.fit_transform(texts)
            y = np.array(labels)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train logistic regression model
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Model accuracy for {goal}: {accuracy:.3f}")
            
            # Save model and vectorizer
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
            os.makedirs(model_dir, exist_ok=True)
            
            model_file = os.path.join(model_dir, f"{goal.lower().replace(' ', '_')}_model.pkl")
            vectorizer_file = os.path.join(model_dir, f"{goal.lower().replace(' ', '_')}_vectorizer.pkl")
            
            logger.info(f"Saving model to {model_file}")
            logger.info(f"Saving vectorizer to {vectorizer_file}")
            
            try:
                joblib.dump(model, model_file)
                joblib.dump(vectorizer, vectorizer_file)
                logger.info(f"Model and vectorizer saved successfully for {goal}")
            except Exception as e:
                logger.error(f"Error saving model for {goal}: {str(e)}")
                raise
            
            # Store in memory
            self.models[goal] = model
            self.vectorizers[goal] = vectorizer
            
            logger.info(f"Model trained and saved for {goal}")
            
        except Exception as e:
            logger.error(f"Error training model for {goal}: {str(e)}")
            raise
    
    def _extract_skills_from_text(self, text):
        """Extract skills mentioned in the resume text"""
        text_lower = text.lower()
        found_skills = []
        
        # Get all possible skills
        all_skills = set()
        for skills_list in self.goals_skills.values():
            all_skills.update(skills_list)
        
        # Skill variations and related terms
        skill_variations = {
            "Python": ["python", "py", "pythonic"],
            "Java": ["java", "javac", "jvm"],
            "Machine Learning": ["ml", "machine learning", "ml engineer", "ml engineer"],
            "Deep Learning": ["deep learning", "neural network", "nn", "dl"],
            "NumPy": ["numpy", "np", "numerical python"],
            "Pandas": ["pandas", "pd", "dataframe"],
            "Scikit-learn": ["scikit-learn", "sklearn", "machine learning library"],
            "TensorFlow": ["tensorflow", "tf", "tensorflow2"],
            "PyTorch": ["pytorch", "torch", "pytorch lightning"],
            "Statistics": ["statistics", "statistical", "statistical analysis"],
            "Linear Algebra": ["linear algebra", "matrix operations", "vector space"],
            "Data Analysis": ["data analysis", "data analytics", "data scientist"],
            "Feature Engineering": ["feature engineering", "feature extraction", "feature selection"],
            "Model Evaluation": ["model evaluation", "model validation", "performance metrics"],
            "Computer Vision": ["computer vision", "cv", "image processing"],
            "Natural Language Processing": ["nlp", "natural language processing", "text analysis"]
        }
        
        for skill in all_skills:
            # Check for exact match
            if skill.lower() in text_lower:
                found_skills.append(skill)
                continue
            
            # Check for variations
            if skill in skill_variations:
                for variation in skill_variations[skill]:
                    if variation in text_lower:
                        found_skills.append(skill)
                        break
            
            # Check for related terms
            if "data structures" in text_lower:
                found_skills.extend(["Data Structures", "Algorithms"])  # Related skills
            if "dsa" in text_lower:
                found_skills.extend(["Data Structures", "Algorithms", "System Design"])  # Related skills
            if "system design" in text_lower:
                found_skills.extend(["System Design", "Architecture"])  # Related skills
            if "sql" in text_lower:
                found_skills.extend(["SQL", "Database Design"])  # Related skills
            if "api" in text_lower:
                found_skills.extend(["REST APIs", "Web Services"])  # Related skills
            if "flask" in text_lower:
                found_skills.extend(["Python", "Web Development"])  # Related skills
            if "leetcode" in text_lower:
                found_skills.extend(["Data Structures", "Algorithms", "Problem Solving"])  # Related skills
        
        return list(set(found_skills))  # Remove duplicates
    
    def _generate_learning_path(self, missing_skills, goal):
        """Generate learning path based on missing skills"""
        learning_path = []
        
        # Define learning recommendations for common skills
        skill_recommendations = {
            "Java": "Master Java fundamentals, OOP concepts, and Spring framework",
            "Python": "Learn Python basics, data structures, and popular libraries",
            "Data Structures": "Study arrays, linked lists, trees, graphs, and hash tables",
            "Algorithms": "Practice sorting, searching, dynamic programming, and graph algorithms",
            "System Design": "Learn system design principles, scalability, and distributed systems",
            "SQL": "Master database concepts, joins, indexing, and query optimization",
            "REST APIs": "Understand REST principles, HTTP methods, and API design",
            "Git": "Learn version control, branching, merging, and collaboration workflows",
            "Machine Learning": "Study ML algorithms, model evaluation, and feature engineering",
            "Deep Learning": "Learn neural networks, CNNs, RNNs, and deep learning frameworks",
            "NumPy": "Master numerical computing and array operations in Python",
            "Pandas": "Learn data manipulation and analysis with Pandas library",
            "Scikit-learn": "Practice machine learning algorithms and model building",
            "TensorFlow": "Build and train neural networks with TensorFlow framework",
            "Statistics": "Learn statistical concepts, hypothesis testing, and probability",
            "Digital Electronics": "Study logic gates, combinational and sequential circuits",
            "Signal Processing": "Learn signal analysis, filtering, and frequency domain concepts",
            "Communication Systems": "Study modulation, channel coding, and wireless systems",
            "Control Systems": "Learn feedback systems, stability analysis, and controller design",
            "Electromagnetics": "Study electromagnetic fields, waves, and antenna theory",
            "Network Theory": "Learn circuit analysis, network theorems, and filter design"
        }
        
        for skill in missing_skills:
            if skill in skill_recommendations:
                learning_path.append(skill_recommendations[skill])
            else:
                learning_path.append(f"Learn {skill} fundamentals and best practices")
        
        # Add goal-specific recommendations
        if goal == "Amazon SDE" and missing_skills:
            learning_path.append("Practice coding problems on LeetCode and HackerRank")
            learning_path.append("Build projects demonstrating system design skills")
        elif goal == "ML Internship" and missing_skills:
            learning_path.append("Complete online ML courses and build portfolio projects")
            learning_path.append("Participate in Kaggle competitions for hands-on experience")
        elif goal == "GATE ECE" and missing_skills:
            learning_path.append("Solve previous year GATE questions and mock tests")
            learning_path.append("Focus on mathematical concepts and problem-solving techniques")
        
        return learning_path[:5]  # Limit to top 5 recommendations
    
    def score_resume(self, resume_text, goal):
        """Score a resume against a specific goal"""
        # Validate goal
        if goal not in self.goals_skills:
            raise ValueError(f"Unsupported goal: {goal}. Supported goals: {list(self.goals_skills.keys())}")

        # Get required skills for the goal
        required_skills = self.goals_skills[goal]
        
        # Extract skills from resume
        found_skills = self._extract_skills_from_text(resume_text)
        
        # Find matched and missing skills
        matched_skills = [skill for skill in required_skills if skill in found_skills]
        missing_skills = [skill for skill in required_skills if skill not in found_skills]
        
        # Calculate score based on skill matching
        # Each skill contributes 100% / number of required skills
        skill_score = len(matched_skills) / len(required_skills)
        
        # Add bonus points for having more than half the skills
        if len(matched_skills) > len(required_skills) / 2:
            skill_score += 0.1  # 10% bonus
        
        # Add bonus points for having all core skills
        core_skills = {
            "ML Internship": ["Python", "Machine Learning", "Deep Learning", "NumPy", "Pandas"],
            "Amazon SDE": ["Java", "Python", "Data Structures", "System Design", "SQL"],
            "GATE ECE": ["Digital Electronics", "Signal Processing", "Control Systems"]
        }
        
        if goal in core_skills and all(skill in found_skills for skill in core_skills[goal]):
            skill_score += 0.2  # 20% bonus for core skills
        
        # Ensure score is between 0 and 1
        skill_score = min(1.0, max(0.0, skill_score))
        
        # Generate learning path
        suggested_learning_path = self._generate_learning_path(missing_skills, goal)
        
        return {
            "score": round(float(skill_score), 3),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "suggested_learning_path": suggested_learning_path
        }

