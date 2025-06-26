from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import json
import os
from pathlib import Path
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, validator
from pydantic.typing import AnyCallable
import uvicorn

from .scorer import ResumeScorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
config = {
    "model_dir": Path(__file__).parent / "model",
    "log_dir": Path(__file__).parent / "logs"
}

logger.info("Application logging initialized")

# Load configuration
def load_application_settings() -> Dict[str, Any]:
    """Load and validate application settings from configuration file"""
    try:
        with open('config.json', 'r') as config_file:
            settings = json.load(config_file)
            
        # Validate required fields
        required_fields = ['version', 'minimum_score_to_pass', 'model_goals_supported']
        for field in required_fields:
            if field not in settings:
                logger.error(f"Missing required configuration field: {field}")
                raise HTTPException(status_code=500, detail=f"Missing required configuration field: {field}")
                
        # Validate types
        if not isinstance(settings['minimum_score_to_pass'], (int, float)):
            raise HTTPException(status_code=500, detail="minimum_score_to_pass must be a number")
            
        if not isinstance(settings['model_goals_supported'], list):
            raise HTTPException(status_code=500, detail="model_goals_supported must be a list")
            
        logger.info("Application settings loaded and validated successfully")
        return settings
    except FileNotFoundError:
        logger.error("Configuration file not found")
        raise HTTPException(status_code=500, detail="Required configuration file is missing")
    except json.JSONDecodeError:
        logger.error("Invalid JSON format in configuration")
        raise HTTPException(status_code=500, detail="Invalid configuration file format")
    except HTTPException as e:
        logger.error(f"Configuration validation error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load configuration")

# Initialize FastAPI application
app = FastAPI(
    title="Resume Evaluation System",
    description="AI-driven resume analysis and scoring service",
    version="1.0.0",
    docs_url=None,  # Disable OpenAPI docs
    redoc_url=None,  # Disable ReDoc
    openapi_url=None  # Disable OpenAPI JSON
)

# Add static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize application components
try:
    app_settings: Dict[str, Any] = load_application_settings()
    scorer: ResumeScorer = ResumeScorer(app_settings)
    logger.info("Application initialized successfully")
except Exception as e:
    logger.error(f"Application initialization failed: {str(e)}")
    app_settings = None
    scorer = None
    raise HTTPException(status_code=500, detail="Service initialization failed")

# Pydantic models
class ResumeRequest(BaseModel):
    """Request model for resume scoring endpoint"""
    student_id: str
    goal: str
    resume_text: str
    
    @validator('student_id')
    def validate_student_id(cls, v: str) -> str:
        """Validate student_id is not empty"""
        if not v.strip():
            raise ValueError('student_id cannot be empty')
        return v.strip()
    
    @validator('goal')
    def validate_goal(cls, v: str, values: Dict[str, Any], **kwargs: Any) -> str:
        """Validate goal is one of the supported values"""
        valid_goals = ['Amazon SDE', 'ML Internship', 'GATE ECE']
        if v not in valid_goals:
            raise ValueError(f'Invalid goal. Must be one of: {", ".join(valid_goals)}')
        return v.strip()
    
    @validator('resume_text')
    def validate_resume_text(cls, v: str) -> str:
        """Validate resume text is not empty and within length limits"""
        if not v.strip():
            raise ValueError('Resume text cannot be empty')
        if len(v) > 10000:  # Limit to 10,000 characters
            raise ValueError('Resume text is too long (max 10,000 characters)')
        return v.strip()

class ResumeResponse(BaseModel):
    """Response model for resume scoring endpoint"""
    score: float
    matched_skills: List[str]
    missing_skills: List[str]
    suggested_learning_path: List[str]

class HealthResponse(BaseModel):
    status: str

class VersionResponse(BaseModel):
    version: str
    supported_goals: List[str]
    minimum_score_threshold: float

# API Endpoints
@app.get("/")
async def get_frontend():
    """Serve the frontend interface"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resume Scorer</title>
    <style>
        /* Base Styles */
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .gradient-bg { 
            background: linear-gradient(to right, #667eea, #764ba2);
            color: white;
            padding: 1.5rem 0;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .container {
            width: 100%;
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 1rem;
        }
        .card {
            background: white;
            border-radius: 0.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 2rem;
            margin: 2rem auto;
            max-width: 48rem;
        }
        .form-group {
            margin-bottom: 1.5rem;
        }
        label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 500;
            color: #374151;
        }
        input[type="text"],
        select,
        textarea {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid #d1d5db;
            border-radius: 0.375rem;
            font-size: 1rem;
        }
        textarea {
            min-height: 8rem;
            resize: vertical;
        }
        button {
            background: linear-gradient(to right, #10b981, #3b82f6);
            color: white;
            font-weight: 600;
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 0.375rem;
            cursor: pointer;
            font-size: 1rem;
            width: 100%;
        }
        button:hover {
            opacity: 0.9;
        }
        .results {
            margin-top: 2rem;
            display: none;
        }
        .skill-tag {
            display: inline-block;
            background: #e0f2fe;
            color: #0369a1;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            margin: 0.25rem;
            font-size: 0.875rem;
        }
        .missing-skill {
            background: #fee2e2;
            color: #b91c1c;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 1rem 0;
        }
        .error-message {
            color: #dc2626;
            margin: 1rem 0;
            padding: 1rem;
            background: #fef2f2;
            border-radius: 0.375rem;
            display: none;
        }
    </style>
</head>
<body>
    <header class="gradient-bg">
        <div class="container">
            <h1 style="font-size: 2.25rem; font-weight: 700; margin: 0;">🚀 Resume Scorer</h1>
            <p style="font-size: 1.125rem; margin: 0.5rem 0 0 0;">AI-Powered Resume Evaluation</p>
        </div>
    </header>

    <main class="container">
        <div class="card">
            <h2 style="font-size: 1.5rem; font-weight: 700; color: #1f2937; margin-top: 0; margin-bottom: 1.5rem;">
                📄 Evaluate Your Resume
            </h2>

            <form id="resumeForm">
                <div class="form-group">
                    <label for="studentId">Student ID</label>
                    <input type="text" id="studentId" required>
                </div>

                <div class="form-group">
                    <label for="goal">Target Goal</label>
                    <select id="goal" required>
                        <option value="">Select your goal</option>
                        <option value="Amazon SDE">Amazon SDE</option>
                        <option value="ML Internship">ML Internship</option>
                        <option value="GATE ECE">GATE ECE</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="resumeText">Resume Text</label>
                    <textarea id="resumeText" required placeholder="Paste your full resume content here..."></textarea>
                </div>

                <button type="submit" id="submitBtn">
                    <span id="submitText">✨ Score My Resume</span>
                    <span id="loadingText" style="display: none;">⏳ Scoring...</span>
                </button>
            </form>

            <div id="error" class="error-message"></div>

            <div id="results" class="results">
                <div style="text-align: center; margin-bottom: 1.5rem;">
                    <h3 style="font-size: 1.25rem; font-weight: 600; margin: 0 0 0.5rem 0;">
                        Score: <span id="score" style="color: #1d4ed8; font-size: 1.5rem;"></span>
                    </h3>
                    <p id="scoreMessage"></p>
                </div>

                <div style="margin-bottom: 1.5rem;">
                    <h4 style="font-size: 1.125rem; font-weight: 700; margin: 0 0 0.5rem 0; color: #1f2937;">
                        ✅ Matched Skills:
                    </h4>
                    <div id="matchedSkills"></div>
                </div>

                <div style="margin-bottom: 1.5rem;">
                    <h4 style="font-size: 1.125rem; font-weight: 700; margin: 0 0 0.5rem 0; color: #1f2937;">
                        ❌ Missing Skills:
                    </h4>
                    <div id="missingSkills"></div>
                </div>

                <div>
                    <h4 style="font-size: 1.125rem; font-weight: 700; margin: 0 0 0.5rem 0; color: #1f2937;">
                        🎓 Suggested Learning Path:
                    </h4>
                    <ul id="learningPath" style="margin: 0; padding-left: 1.5rem;"></ul>
                </div>
            </div>
        </div>
    </main>

    <footer style="text-align: center; color: #9ca3af; font-size: 0.875rem; padding: 1.5rem 0;">
        Built with ❤️ by Rohini - Team Insight Architects
    </footer>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const form = document.getElementById('resumeForm');
            const submitBtn = document.getElementById('submitBtn');
            const submitText = document.getElementById('submitText');
            const loadingText = document.getElementById('loadingText');
            const errorDiv = document.getElementById('error');
            const resultsDiv = document.getElementById('results');
            const scoreSpan = document.getElementById('score');
            const scoreMessage = document.getElementById('scoreMessage');
            const matchedSkillsDiv = document.getElementById('matchedSkills');
            const missingSkillsDiv = document.getElementById('missingSkills');
            const learningPathUl = document.getElementById('learningPath');

            form.addEventListener('submit', async function(e) {
                e.preventDefault();
                
                // Show loading state
                submitText.style.display = 'none';
                loadingText.style.display = 'inline';
                errorDiv.style.display = 'none';
                resultsDiv.style.display = 'none';
                submitBtn.disabled = true;

                const formData = {
                    student_id: document.getElementById('studentId').value,
                    goal: document.getElementById('goal').value,
                    resume_text: document.getElementById('resumeText').value
                };

                try {
                    const response = await fetch('/score', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(formData)
                    });

                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || 'Error occurred');
                    }

                    const results = await response.json();
                    displayResults(results);
                } catch (err) {
                    showError(err.message);
                } finally {
                    submitText.style.display = 'inline';
                    loadingText.style.display = 'none';
                    submitBtn.disabled = false;
                }
            });

            function displayResults(results) {
                // Update score
                scoreSpan.textContent = Math.round(results.score * 100) + '%';
                
                // Update score message
                if (results.score >= 0.7) {
                    scoreMessage.textContent = '🌟 Excellent match! Ready to apply!';
                } else if (results.score >= 0.6) {
                    scoreMessage.textContent = '💡 Good match! Work on missing skills.';
                } else {
                    scoreMessage.textContent = '📚 Needs improvement. Focus on learning path.';
                }

                // Update matched skills
                matchedSkillsDiv.innerHTML = '';
                results.matched_skills.forEach(skill => {
                    const skillSpan = document.createElement('span');
                    skillSpan.className = 'skill-tag';
                    skillSpan.textContent = '🎯 ' + skill;
                    matchedSkillsDiv.appendChild(skillSpan);
                });

                // Update missing skills
                missingSkillsDiv.innerHTML = '';
                results.missing_skills.forEach(skill => {
                    const skillSpan = document.createElement('span');
                    skillSpan.className = 'skill-tag missing-skill';
                    skillSpan.textContent = '⚠️ ' + skill;
                    missingSkillsDiv.appendChild(skillSpan);
                });

                // Update learning path
                learningPathUl.innerHTML = '';
                results.suggested_learning_path.forEach(item => {
                    const li = document.createElement('li');
                    li.textContent = item;
                    li.style.marginBottom = '0.5rem';
                    learningPathUl.appendChild(li);
                });

                // Show results
                resultsDiv.style.display = 'block';
            }

            function showError(message) {
                errorDiv.textContent = '❌ ' + message;
                errorDiv.style.display = 'block';
            }
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api-docs")
async def get_docs():
    """Redirect to root"""
    return RedirectResponse(url="/")

@app.get("/api-redoc")
async def get_redoc():
    """Redirect to root"""
    return RedirectResponse(url="/")

@app.post("/score", response_model=ResumeResponse)
async def score_resume(request: ResumeRequest):
    """Score a resume against a target goal"""
    if scorer is None:
        raise HTTPException(status_code=500, detail="Service not properly initialized")
    
    try:
        if config and config.get("log_score_details", False):
            logger.info(f"Scoring resume for student {request.student_id} with goal {request.goal}")
        
        result = scorer.score_resume(
            resume_text=request.resume_text,
            goal=request.goal
        )
        
        return ResumeResponse(**result)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error scoring resume: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if scorer is None or config is None:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    return HealthResponse(status="ok")

@app.get("/version", response_model=VersionResponse)
async def get_version():
    """Get version and configuration information"""
    if config is None:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    return VersionResponse(
        version=config.get("version", "1.0.0"),
        supported_goals=config.get("model_goals_supported", []),
        minimum_score_threshold=config.get("minimum_score_to_pass", 0.6)
    )

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)