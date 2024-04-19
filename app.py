
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional

from loan_status.components import APP_HOST, APP_PORT
from loan_status.pipeline.prediction_pipeline import LoanStatusData, LoanStatusClassifier
from loan_status.pipeline.training_pipeline import TrainPipeline

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    def __init__(self, request: Request):
        self.Gender: Request = request
        self.Married: Optional[str] = None
        self.Dependents: Optional[str] = None
        self.Education: Optional[str] = None
        self.Self_Employed: Optional[str] = None
        self.ApplicantIncome: Optional[str] = None
        self.CoapplicantIncome: Optional[str] = None
        self.LoanAmount: Optional[str] = None
        self.Loan_Amount_Term: Optional[str] = None
        self.Credit_History: Optional[str] = None
        self.Property_Area: Optional[str] = None
        

    async def get_loanstatus_data(self):
        form = await self.request.form()
        self.Gender = form.get("Gender")
        self.Married = form.get("Married")
        self.Dependents = form.get("Dependents")
        self.Education = form.get("Education")
        self.Self_Employed = form.get("Self_Employed")
        self.ApplicantIncome = form.get("ApplicantIncome")
        self.CoapplicantIncome = form.get("CoapplicantIncome")
        self.LoanAmount = form.get("LoanAmount")
        self.Loan_Amount_Term = form.get("Loan_Amount_Term")
        self.Credit_History= form.get("Credit_History")
        self.Property_Area = form.get("Property_Area")

@app.get("/", tags=["authentication"])
async def index(request: Request):

    return templates.TemplateResponse(
            "loanstatus.html",{"request": request, "context": "Rendering"})


@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/")
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        await form.get_loanstatus_data()
        
        loanstatus_data = LoanStatusData(
                                Gender= form.Gender,
                                Married = form.Married,
                                Dependents = form.Dependents,
                                Education = form.Education,
                                Self_Employed= form.Self_Employed,
                                ApplicantIncome= form.ApplicantIncome,
                                CoapplicantIncome = form.CoapplicantIncome,
                                LoanAmount= form.LoanAmount,
                                Loan_Amount_Term = form.Loan_Amount_Term,
                                Credit_History = form.Credit_History,
                                Property_Area = form.Property_Area
                                )
        
        loanstatus_df = loanstatus_data.get_loanstatus_input_data_frame()

        model_predictor = LoanStatusClassifier()

        value = model_predictor.predict(dataframe=loanstatus_df)[0]

        status = None
        if value == 0:
            status = "loan-approved"
        else:
            status = "loan Not-Approved"

        return templates.TemplateResponse(
            "loanstatus.html",
            {"request": request, "context": status},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)