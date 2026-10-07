# AI Health Risk Predictor - Standard Operating Procedure (SOP)

## Document Information
- **Project Name:** AI Health Risk Predictor
- **Version:** 1.0
- **Last Updated:** October 2026
- **Purpose:** Complete operational guide for project development, deployment, and maintenance

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Prerequisites & Environment Setup](#prerequisites--environment-setup)
3. [Development Workflow](#development-workflow)
4. [Backend Setup & Configuration](#backend-setup--configuration)
5. [Frontend Setup & Configuration](#frontend-setup--configuration)
6. [Machine Learning Pipeline](#machine-learning-pipeline)
7. [API Integration](#api-integration)
8. [Testing Procedures](#testing-procedures)
9. [Deployment Guide](#deployment-guide)
10. [Maintenance & Monitoring](#maintenance--monitoring)
11. [Troubleshooting Guide](#troubleshooting-guide)
12. [Team Roles & Responsibilities](#team-roles--responsibilities)

---

## 1. Project Overview

### Project Objective
Build an AI-powered healthcare application that provides educational risk estimates for common health conditions (Type 2 Diabetes, Cardiovascular Disease, Hypertension).

### Key Deliverables
- React.js frontend with healthcare UI
- Python FastAPI backend
- ML model for risk prediction
- Complete documentation
- Test coverage
- Deployment-ready application

### Technology Stack
| Component | Technology |
|-----------|-----------|
| Frontend | React.js + Tailwind CSS |
| Backend | Python FastAPI |
| ML Framework | scikit-learn |
| Database | Firebase/MongoDB |
| Charts | Recharts |
| Deployment | Docker + Cloud Platform |

---

## 2. Prerequisites & Environment Setup

### 2.1 System Requirements
- **OS:** Windows 10+, macOS 10.14+, or Linux (Ubuntu 20.04+)
- **RAM:** 8GB minimum
- **Disk Space:** 10GB free
- **Internet:** Required for dependencies and API calls

### 2.2 Required Software

#### Node.js & npm (Frontend)
```bash
# Check if installed
node --version  # Should be v16.0.0 or higher
npm --version   # Should be v7.0.0 or higher

# Download from: https://nodejs.org/
```

#### Python (Backend & ML)
```bash
# Check if installed
python --version  # Should be Python 3.8 or higher
pip --version

# Download from: https://www.python.org/downloads/
```

#### Git (Version Control)
```bash
git --version  # Should be 2.25.0 or higher
```

#### Docker (Optional for Deployment)
```bash
docker --version
docker-compose --version
```

### 2.3 IDE & Tools Recommended
- **VSCode** with Python & JavaScript extensions
- **PyCharm Community Edition** (optional)
- **Postman** for API testing
- **DBeaver** for database management

### 2.4 Environment Variables Setup

Create `.env.example` file in project root:
```
# Backend Configuration
BACKEND_URL=http://localhost:8000
PYTHONPATH=./backend

# Frontend Configuration
REACT_APP_API_URL=http://localhost:8000

# Database
FIREBASE_API_KEY=your_firebase_key
FIREBASE_AUTH_DOMAIN=your_domain.firebaseapp.com
FIREBASE_PROJECT_ID=your_project_id
FIREBASE_STORAGE_BUCKET=your_bucket.appspot.com

# Alternative: MongoDB
MONGODB_URI=mongodb://localhost:27017/health_risk_predictor
MONGODB_USER=admin
MONGODB_PASSWORD=password

# ML Model
MODEL_PATH=./backend/models/risk_predictor_model.pkl
DATASET_PATH=./backend/datasets/

# Security
JWT_SECRET=your_secret_key_here
ENVIRONMENT=development

# Email (Optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
```

---

## 3. Development Workflow

### 3.1 Folder Structure

```
AI-Health-Risk-Predictor/
│
├── frontend/                 # React application
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Home.jsx
│   │   │   ├── HealthForm.jsx
│   │   │   ├── ResultDashboard.jsx
│   │   │   ├── Dashboard.jsx
│   │   │   ├── About.jsx
│   │   │   └── PrivacyDisclaimer.jsx
│   │   ├── pages/
│   │   ├── styles/
│   │   ├── services/
│   │   ├── utils/
│   │   ├── App.jsx
│   │   └── index.js
│   ├── .env
│   ├── tailwind.config.js
│   ├── package.json
│   └── README.md
│
├── backend/                  # FastAPI application
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── models.py         # Database models
│   │   ├── schemas.py        # Pydantic schemas
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── predictions.py
│   │   │   ├── assessments.py
│   │   │   └── health.py
│   │   └── utils/
│   │       └── validators.py
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── preprocessing.py
│   │   ├── evaluation.py
│   │   └── inference.py
│   ├── datasets/
│   │   └── data_loader.py
│   ├── models/
│   │   └── risk_predictor_model.pkl
│   ├── tests/
│   ├── requirements.txt
│   ├── .env
│   └── README.md
│
├── ml/                       # ML pipeline & training
│   ├── notebooks/
│   │   └── model_training.ipynb
│   ├── data/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── prepare_dataset.py
│   ├── models/
│   ├── scripts/
│   │   ├── train_model.py
│   │   ├── evaluate_model.py
│   │   └── prepare_data.py
│   └── README.md
│
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
│
├── docs/
│   ├── API_DOCUMENTATION.md
│   ├── DEPLOYMENT_GUIDE.md
│   └── ARCHITECTURE.md
│
├── .gitignore
├── README.md
├── SOP.md (this file)
└── CONTRIBUTING.md
```

### 3.2 Git Workflow

```bash
# Initialize repository
git init
git remote add origin <repository-url>

# Create main branches
git branch develop
git branch production
git checkout develop

# Feature branch naming convention
# feature/<feature-name>
# bugfix/<bug-name>
# hotfix/<critical-fix>

# Example: Create feature branch
git checkout -b feature/health-form-validation
git add .
git commit -m "Add form validation for health inputs"
git push origin feature/health-form-validation

# Create Pull Request on GitHub/GitLab
# After review and approval, merge to develop
# Create Release PR from develop to production
```

### 3.3 Development Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd AI-Health-Risk-Predictor

# Copy environment file
cp .env.example .env

# Edit .env with your values
nano .env

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

---

## 4. Backend Setup & Configuration

### 4.1 Python Virtual Environment

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### 4.2 Requirements.txt Content

Create `backend/requirements.txt`:
```
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
python-dotenv==1.0.0
sqlalchemy==2.0.23
pymongo==4.6.0
firebase-admin==6.2.0
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.26.2
joblib==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
pytest==7.4.3
pytest-cov==4.1.0
python-multipart==0.0.6
cors==1.0.1
aiofiles==23.2.1
requests==2.31.0
```

### 4.3 Database Configuration

#### Option A: Firebase Setup
```bash
# 1. Create Firebase project at https://firebase.google.com/
# 2. Download service account key JSON
# 3. Place in backend/config/firebase-key.json
# 4. Update FIREBASE_* environment variables

# 5. Install Firebase
pip install firebase-admin
```

#### Option B: MongoDB Setup
```bash
# 1. Install MongoDB locally or use MongoDB Atlas (cloud)
# 2. Create database named 'health_risk_predictor'
# 3. Create collections:
#    - assessments
#    - predictions
#    - users (optional)

# 4. Update MONGODB_URI in .env

# 5. Connection string format:
# mongodb://username:password@host:port/database_name
```

### 4.4 FastAPI Application Structure

Create `backend/app/main.py`:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="AI Health Risk Predictor API",
    description="API for health risk assessment and prediction",
    version="1.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "http://localhost:3000")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers
from app.routes import predictions, assessments, health

app.include_router(predictions.router, prefix="/api/v1/predictions", tags=["Predictions"])
app.include_router(assessments.router, prefix="/api/v1/assessments", tags=["Assessments"])
app.include_router(health.router, prefix="/api/v1/health", tags=["Health"])

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "AI Health Risk Predictor API",
        "version": "1.0.0"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 4.5 Run Backend Server

```bash
# Start development server
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
python -m uvicorn app.main:app --reload

# Server runs on http://localhost:8000
# API docs available at http://localhost:8000/docs
```

---

## 5. Frontend Setup & Configuration

### 5.1 React Project Setup

```bash
# Navigate to project root
cd AI-Health-Risk-Predictor

# Create React app with Vite (faster)
npm create vite@latest frontend -- --template react

# Or with Create React App
npx create-react-app frontend

# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Install Tailwind CSS
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p

# Install Recharts for charts
npm install recharts

# Install Axios for API calls
npm install axios

# Install React Router for navigation
npm install react-router-dom

# Install additional utilities
npm install classnames date-fns
```

### 5.2 Tailwind Configuration

Create `frontend/tailwind.config.js`:
```javascript
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        healthcare: {
          primary: '#0066CC',
          secondary: '#00CC99',
          danger: '#FF3333',
          warning: '#FFAA00',
          light: '#F5F7FA',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
```

### 5.3 Create React Components Structure

```bash
# Create component directories
mkdir -p src/components
mkdir -p src/pages
mkdir -p src/services
mkdir -p src/utils
mkdir -p src/styles

# Create components
touch src/components/Home.jsx
touch src/components/HealthForm.jsx
touch src/components/ResultDashboard.jsx
touch src/components/Dashboard.jsx
touch src/components/About.jsx
touch src/components/PrivacyDisclaimer.jsx

# Create services
touch src/services/apiService.js
touch src/services/predictionService.js

# Create utilities
touch src/utils/validators.js
touch src/utils/formatters.js
```

### 5.4 API Service Setup

Create `frontend/src/services/apiService.js`:
```javascript
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const apiService = axios.create({
  baseURL: `${API_URL}/api/v1`,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for auth token (if needed)
apiService.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Add response interceptor for error handling
apiService.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

export default apiService;
```

### 5.5 Run Frontend Server

```bash
# Navigate to frontend directory
cd frontend

# Start development server
npm start

# Frontend runs on http://localhost:3000
# Ensure backend is running on http://localhost:8000
```

---

## 6. Machine Learning Pipeline

### 6.1 Dataset Preparation

Create `backend/ml/data/prepare_dataset.py`:

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

def load_dataset(filepath):
    """Load and return dataset"""
    df = pd.read_csv(filepath)
    return df

def handle_missing_values(df):
    """Handle missing values in dataset"""
    # Drop rows where critical features are missing
    critical_features = ['age', 'blood_pressure', 'blood_glucose']
    df = df.dropna(subset=critical_features)
    
    # Fill other missing values with median
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    
    return df

def preprocess_features(df):
    """Preprocess categorical and numerical features"""
    
    # Categorical features
    categorical_cols = ['biological_sex', 'smoking_status', 'family_history']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # Numerical features normalization
    scaler = StandardScaler()
    numeric_cols = ['age', 'height', 'weight', 'blood_pressure', 
                    'blood_glucose', 'cholesterol']
    
    available_numeric = [col for col in numeric_cols if col in df.columns]
    df[available_numeric] = scaler.fit_transform(df[available_numeric])
    
    return df, scaler, label_encoders

def create_target_variable(df):
    """Create risk category target variable"""
    # Example: Create based on multiple conditions
    def categorize_risk(row):
        risk_score = 0
        
        if row.get('blood_glucose', 0) > 125:
            risk_score += 2
        if row.get('blood_pressure', 0) > 140:
            risk_score += 2
        if row.get('cholesterol', 0) > 200:
            risk_score += 1
        if row.get('smoking_status', 0) == 1:
            risk_score += 1
        if row.get('physical_activity', 0) < 30:
            risk_score += 1
        if row.get('family_history', 0) == 1:
            risk_score += 1
        if row.get('age', 0) > 50:
            risk_score += 1
        
        if risk_score <= 2:
            return 'Low'
        elif risk_score <= 4:
            return 'Moderate'
        else:
            return 'High'
    
    df['risk_category'] = df.apply(categorize_risk, axis=1)
    return df

def prepare_data(input_filepath, output_dir):
    """Complete data preparation pipeline"""
    
    # Load data
    df = load_dataset(input_filepath)
    print(f"Loaded dataset with shape: {df.shape}")
    
    # Handle missing values
    df = handle_missing_values(df)
    print(f"After missing value handling: {df.shape}")
    
    # Create target variable
    df = create_target_variable(df)
    
    # Preprocess features
    df, scaler, label_encoders = preprocess_features(df)
    
    # Split features and target
    X = df.drop('risk_category', axis=1)
    y = df['risk_category']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save preprocessed data
    joblib.dump({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'label_encoders': label_encoders
    }, f'{output_dir}/preprocessed_data.pkl')
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler, label_encoders

if __name__ == "__main__":
    prepare_data('./data/raw/health_data.csv', './data/processed')
```

### 6.2 Model Training

Create `backend/ml/scripts/train_model.py`:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import pandas as pd

def train_model(X_train, y_train):
    """Train the risk prediction model"""
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("Model trained successfully!")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\n=== Model Evaluation Metrics ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def get_feature_importance(model, feature_names):
    """Get feature importance"""
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== Top 10 Important Features ===")
    print(importance_df.head(10))
    
    return importance_df

def save_model(model, filepath):
    """Save trained model"""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

if __name__ == "__main__":
    # Load preprocessed data
    data = joblib.load('./data/processed/preprocessed_data.pkl')
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Get feature importance
    get_feature_importance(model, X_train.columns)
    
    # Save model
    save_model(model, './models/risk_predictor_model.pkl')
```

### 6.3 Model Inference

Create `backend/ml/inference.py`:

```python
import joblib
import numpy as np
import pandas as pd

class HealthRiskPredictor:
    def __init__(self, model_path, scaler_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
    
    def predict(self, input_data):
        """Make prediction on input data"""
        
        # Scale features
        scaled_data = self.scaler.transform([input_data])
        
        # Get prediction
        prediction = self.model.predict(scaled_data)[0]
        probabilities = self.model.predict_proba(scaled_data)[0]
        
        # Map predictions to risk categories
        risk_categories = self.model.classes_
        confidence = max(probabilities) * 100
        
        return {
            'risk_category': prediction,
            'confidence': confidence,
            'probabilities': {
                risk: float(prob) * 100 
                for risk, prob in zip(risk_categories, probabilities)
            }
        }
    
    def get_influential_factors(self, input_data, top_n=5):
        """Get factors influencing the prediction"""
        
        feature_importance = pd.DataFrame({
            'feature': self.model.feature_names_in_,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return feature_importance.head(top_n).to_dict('records')
```

---

## 7. API Integration

### 7.1 Prediction Endpoint

Create `backend/app/routes/predictions.py`:

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ml.inference import HealthRiskPredictor
import os

router = APIRouter()

# Initialize predictor
MODEL_PATH = os.getenv('MODEL_PATH', './models/risk_predictor_model.pkl')
SCALER_PATH = os.getenv('SCALER_PATH', './models/scaler.pkl')

try:
    predictor = HealthRiskPredictor(MODEL_PATH, SCALER_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    predictor = None

class HealthDataInput(BaseModel):
    age: int
    biological_sex: str
    height: float
    weight: float
    blood_pressure: float
    blood_glucose: float
    cholesterol: float
    smoking_status: bool
    physical_activity: int
    family_history: bool
    sleep_duration: int

@router.post("/")
async def create_prediction(data: HealthDataInput):
    """Create a health risk prediction"""
    
    if not predictor:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Prepare input data
        input_array = [
            data.age,
            1 if data.biological_sex.lower() == 'male' else 0,
            data.height,
            data.weight,
            data.blood_pressure,
            data.blood_glucose,
            data.cholesterol,
            int(data.smoking_status),
            data.physical_activity,
            int(data.family_history),
            data.sleep_duration
        ]
        
        # Make prediction
        result = predictor.predict(input_array)
        
        # Get influential factors
        factors = predictor.get_influential_factors(input_array)
        
        return {
            "status": "success",
            "prediction": result,
            "influential_factors": factors,
            "disclaimer": "This is an educational estimate only, not a medical diagnosis."
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

### 7.2 Assessment Endpoint

Create `backend/app/routes/assessments.py`:

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
import uuid

router = APIRouter()

# In-memory storage (replace with database)
assessments_db = {}

class AssessmentData(BaseModel):
    user_id: str
    health_data: dict
    prediction_result: dict
    notes: str = ""

@router.post("/")
async def create_assessment(assessment: AssessmentData):
    """Store assessment record"""
    
    assessment_id = str(uuid.uuid4())
    
    assessment_record = {
        "id": assessment_id,
        "user_id": assessment.user_id,
        "health_data": assessment.health_data,
        "prediction_result": assessment.prediction_result,
        "notes": assessment.notes,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }
    
    assessments_db[assessment_id] = assessment_record
    
    return assessment_record

@router.get("/{assessment_id}")
async def get_assessment(assessment_id: str):
    """Retrieve assessment by ID"""
    
    if assessment_id not in assessments_db:
        raise HTTPException(status_code=404, detail="Assessment not found")
    
    return assessments_db[assessment_id]

@router.get("/user/{user_id}")
async def get_user_assessments(user_id: str):
    """Get all assessments for a user"""
    
    user_assessments = [
        a for a in assessments_db.values() 
        if a['user_id'] == user_id
    ]
    
    return user_assessments
```

---

## 8. Testing Procedures

### 8.1 Backend Testing

Create `backend/tests/test_predictions.py`:

```python
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_prediction_endpoint():
    """Test prediction endpoint"""
    
    payload = {
        "age": 45,
        "biological_sex": "male",
        "height": 175,
        "weight": 80,
        "blood_pressure": 130,
        "blood_glucose": 110,
        "cholesterol": 200,
        "smoking_status": False,
        "physical_activity": 30,
        "family_history": False,
        "sleep_duration": 7
    }
    
    response = client.post("/api/v1/predictions/", json=payload)
    
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "status" in response.json()

def test_health_endpoint():
    """Test health check endpoint"""
    
    response = client.get("/health")
    
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@pytest.mark.parametrize("age,risk", [
    (25, "Low"),
    (50, "Moderate"),
    (70, "High")
])
def test_age_impact_on_risk(age, risk):
    """Test that age impacts risk prediction"""
    
    payload = {
        "age": age,
        "biological_sex": "male",
        "height": 175,
        "weight": 80,
        "blood_pressure": 120,
        "blood_glucose": 100,
        "cholesterol": 180,
        "smoking_status": False,
        "physical_activity": 60,
        "family_history": False,
        "sleep_duration": 8
    }
    
    response = client.post("/api/v1/predictions/", json=payload)
    
    assert response.status_code == 200
```

### 8.2 Frontend Testing

Create `frontend/src/__tests__/HealthForm.test.js`:

```javascript
import { render, screen, fireEvent } from '@testing-library/react';
import HealthForm from '../components/HealthForm';

describe('HealthForm Component', () => {
  it('renders form fields', () => {
    render(<HealthForm />);
    
    expect(screen.getByLabelText(/age/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/biological sex/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/height/i)).toBeInTheDocument();
  });

  it('validates required fields', () => {
    render(<HealthForm />);
    
    const submitBtn = screen.getByRole('button', { name: /submit/i });
    fireEvent.click(submitBtn);
    
    // Check for validation messages
    expect(screen.getByText(/age is required/i)).toBeInTheDocument();
  });

  it('submits form with valid data', () => {
    const mockSubmit = jest.fn();
    render(<HealthForm onSubmit={mockSubmit} />);
    
    fireEvent.change(screen.getByLabelText(/age/i), { target: { value: '45' } });
    fireEvent.click(screen.getByRole('button', { name: /submit/i }));
    
    expect(mockSubmit).toHaveBeenCalled();
  });
});
```

### 8.3 Run Tests

```bash
# Backend tests
cd backend
pytest tests/ --cov=app --cov-report=html

# Frontend tests
cd frontend
npm test -- --coverage
```

---

## 9. Deployment Guide

### 9.1 Docker Setup

Create `docker/Dockerfile.backend`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY backend/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY backend/ .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Create `docker/Dockerfile.frontend`:

```dockerfile
FROM node:18-alpine AS build

WORKDIR /app

COPY frontend/package*.json ./

RUN npm ci

COPY frontend/ .

RUN npm run build

# Production stage
FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html

COPY docker/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

Create `docker/docker-compose.yml`:

```yaml
version: '3.8'

services:
  backend:
    build:
      context: ..
      dockerfile: docker/Dockerfile.backend
    ports:
      - "8000:8000"
    environment:
      - MONGODB_URI=mongodb://mongo:27017/health_risk_predictor
      - ENVIRONMENT=production
    depends_on:
      - mongo
    volumes:
      - ../backend:/app

  frontend:
    build:
      context: ..
      dockerfile: docker/Dockerfile.frontend
    ports:
      - "80:80"
    depends_on:
      - backend
    environment:
      - REACT_APP_API_URL=http://backend:8000

  mongo:
    image: mongo:6.0
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db

volumes:
  mongo_data:
```

### 9.2 Deployment Steps

#### Local Docker Deployment
```bash
# Build and run with docker-compose
cd docker
docker-compose up --build

# Application available at:
# Frontend: http://localhost
# Backend: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

#### Cloud Deployment (Example: AWS/GCP)

```bash
# 1. Build Docker images
docker build -f docker/Dockerfile.backend -t health-risk-backend:latest .
docker build -f docker/Dockerfile.frontend -t health-risk-frontend:latest .

# 2. Push to Docker Registry
docker tag health-risk-backend:latest your-registry/health-risk-backend:latest
docker push your-registry/health-risk-backend:latest

docker tag health-risk-frontend:latest your-registry/health-risk-frontend:latest
docker push your-registry/health-risk-frontend:latest

# 3. Deploy to Kubernetes (if using K8s)
kubectl apply -f deployment.yaml

# 4. Or deploy to AWS Elastic Beanstalk
eb create health-risk-env
eb deploy
```

### 9.3 Environment Variables for Production

```
# Production .env
ENVIRONMENT=production
BACKEND_URL=https://api.yourdomain.com
FRONTEND_URL=https://yourdomain.com

# Security
JWT_SECRET=your-very-secure-secret-key-here
ALLOWED_ORIGINS=https://yourdomain.com

# Database
MONGODB_URI=mongodb+srv://user:password@cluster.mongodb.net/health_risk_predictor

# Email (for notifications)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USER=your-email@gmail.com
EMAIL_PASSWORD=app-specific-password

# Logging
LOG_LEVEL=info
```

---

## 10. Maintenance & Monitoring

### 10.1 Logging Setup

Create `backend/app/config.py`:

```python
import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logging():
    """Configure application logging"""
    
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logger = logging.getLogger('health_risk_app')
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = RotatingFileHandler(
        f'{log_dir}/app.log',
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
```

### 10.2 Model Monitoring

```python
# Track model performance over time
def log_prediction(input_data, prediction, timestamp):
    """Log prediction for monitoring"""
    
    log_entry = {
        'timestamp': timestamp,
        'input_data': input_data,
        'prediction': prediction,
        'model_version': '1.0'
    }
    
    # Store in database or monitoring service
    # Example: Send to Elasticsearch, Prometheus, etc.
```

### 10.3 Health Check Monitoring

```bash
# Monitor application health
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy"}
```

### 10.4 Regular Maintenance Tasks

| Task | Frequency | Owner |
|------|-----------|-------|
| Review logs | Daily | DevOps |
| Update dependencies | Weekly | Security Team |
| Backup database | Daily | DevOps |
| Model performance check | Weekly | ML Engineer |
| Security audit | Monthly | Security Team |
| User feedback review | Weekly | Product Manager |

---

## 11. Troubleshooting Guide

### 11.1 Common Backend Issues

| Issue | Solution |
|-------|----------|
| **Port 8000 already in use** | `lsof -i :8000` and kill process, or use different port |
| **Module not found error** | Activate virtual environment: `source venv/bin/activate` |
| **CORS errors** | Check `FRONTEND_URL` in `.env` matches frontend URL |
| **Database connection failed** | Verify MongoDB running: `mongosh` or Firebase credentials |
| **Model loading error** | Check `MODEL_PATH` exists and file not corrupted |

### 11.2 Common Frontend Issues

| Issue | Solution |
|-------|----------|
| **Blank screen** | Open browser DevTools (F12), check console for errors |
| **API not responding** | Ensure backend running on `http://localhost:8000` |
| **Styling issues** | Run `npm run build:css` or rebuild Tailwind |
| **CORS errors** | Check backend CORS middleware configuration |
| **Module build failed** | Delete `node_modules` and `package-lock.json`, run `npm install` |

### 11.3 Debug Mode

```bash
# Backend debug logging
export PYTHONUNBUFFERED=1
python -m uvicorn app.main:app --reload --log-level debug

# Frontend debug mode
export DEBUG=* npm start
```

---

## 12. Team Roles & Responsibilities

### 12.1 Role Definitions

| Role | Responsibilities |
|------|------------------|
| **Project Manager** | Timeline, deliverables, stakeholder communication |
| **Frontend Developer** | React components, UI/UX, form validation, API integration |
| **Backend Developer** | FastAPI setup, API endpoints, database integration, security |
| **ML Engineer** | Model training, data preparation, inference pipeline, model evaluation |
| **DevOps Engineer** | Deployment, monitoring, CI/CD pipeline, infrastructure |
| **QA Engineer** | Testing, bug reporting, quality assurance |

### 12.2 Communication Protocol

- **Daily Standup:** 10:00 AM (15 minutes)
- **Sprint Planning:** Monday 2:00 PM (1 hour)
- **Sprint Review:** Friday 3:00 PM (1 hour)
- **Emergency Issues:** Slack immediate notification

### 12.3 Version Control Guidelines

```
Commit message format:
[FEATURE/BUGFIX/DOCS] - Brief description

Example:
[FEATURE] - Add health form validation
[BUGFIX] - Fix CORS error in prediction endpoint
[DOCS] - Update API documentation
```

---

## Appendix A: Quick Start Checklist

- [ ] Clone repository
- [ ] Set up Python virtual environment
- [ ] Install backend dependencies
- [ ] Install Node.js and npm
- [ ] Set up React frontend
- [ ] Configure `.env` file
- [ ] Start MongoDB/Firebase
- [ ] Run backend server
- [ ] Run frontend server
- [ ] Test API endpoints in Postman/Insomnia
- [ ] Run test suites
- [ ] Build Docker images
- [ ] Test Docker deployment

## Appendix B: Useful Commands

```bash
# Backend
cd backend
source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm start
npm test

# ML Model Training
cd backend
python ml/scripts/train_model.py

# Docker
docker-compose up --build
docker-compose down

# Testing
pytest tests/ --cov
npm test -- --coverage

# Deployment
git push origin main  # Triggers CI/CD
```

## Appendix C: Resources

- **FastAPI Documentation:** https://fastapi.tiangolo.com/
- **React Documentation:** https://react.dev/
- **scikit-learn Documentation:** https://scikit-learn.org/
- **Tailwind CSS:** https://tailwindcss.com/
- **Docker Documentation:** https://docs.docker.com/

---

**Document Version:** 1.0  
**Last Updated:** October 2026  
**Next Review Date:** January 2027
