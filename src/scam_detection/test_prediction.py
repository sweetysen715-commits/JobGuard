from predict import predict_job_scam

sample_job = {
    "title": "Python Developer",
    "company_profile": "An established software development company.",
    "description": "We are hiring a Python developer with programming experience.",
    "requirements": "Python, SQL, Flask, problem solving.",
    "benefits": "Health insurance and paid leave.",
    "salary_range": "Not Disclosed",
    "employment_type": "Full-time",
    "required_experience": "Entry level",
    "required_education": "Bachelor's Degree",
    "industry": "Information Technology",
    "function": "Engineering",
    "location": "Kolkata, India",
    "telecommuting": 0,
    "has_company_logo": 1,
    "has_questions": 1
}

result = predict_job_scam(sample_job)

print("\nJOBGUARD SCAM DETECTION")
print("-----------------------")

for key, value in result.items():
    print(f"{key}: {value}")