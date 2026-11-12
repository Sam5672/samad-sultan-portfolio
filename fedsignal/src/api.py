from fastapi import FastAPI
from pydantic import BaseModel
from analyze import analyze_statement

app = FastAPI()


class AnalyzeRequest(BaseModel):
    country: str
    statement: str


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    return analyze_statement(country=req.country, statement=req.statement)
