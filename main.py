
import sys
from pathlib import Path

from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(
    0,
    str(BASE_DIR / "src" / "skill_analysis")
)
sys.path.insert(
    0,
    str(BASE_DIR / "src" / "scam_detection")
)

from predict import predict_job_scam

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "app" / "templates")
)


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    error = None

    if request.method == "POST":

        job_data = request.form.to_dict()

        try:
            result = predict_job_scam(job_data)
        except Exception as e:
            error = f"Prediction failed: {e}"

    return render_template(
        "index.html",
        result=result,
        error=error
    )


@app.route("/health")
def health():
    return {
        "status": "success",
        "message": "JobGuard API is running!"
    }

@app.route("/skills", methods=["GET", "POST"])
def skills():

    result = None
    error = None

    if request.method == "POST":

        resume = request.files.get("resume")
        job_description = request.form.get("job_description", "")

        if not resume or not resume.filename:
            error = "Please upload your resume."

        elif not resume.filename.lower().endswith((".pdf", ".docx")):
            error = "Only PDF and DOCX files are supported."

        elif not job_description.strip():
            error = "Please enter a job description."

        else:
            import tempfile
            from pathlib import Path

            temp_path = None

            try:
                from skill_analyzer import analyze_resume_file

                suffix = Path(resume.filename).suffix.lower()

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp_file:
                    temp_path = Path(temp_file.name)
                    resume.save(temp_path)

                result = analyze_resume_file(
                    str(temp_path),
                    job_description
                )

            except Exception as e:
                error = f"Analysis failed: {e}"

            finally:
                if temp_path and temp_path.exists():
                    temp_path.unlink()

    return render_template(
        "skill_analysis.html",
        result=result,
        error=error
    )
if __name__ == "__main__":
    app.run(debug=True)
