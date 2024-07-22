import uvicorn
import helper
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class RequestData(BaseModel):
    abstract: str
    words: int
    papers: int

class ResponseData(BaseModel):
    summary: str
    ids: List[str]

base_prompt = "You will be provided with an abstract of a scientific document and other references papers in triple quotes. Your task is to write the related work section of the document using only the provided abstracts and other references papers. Please write the related work section creating a cohesive storyline by doing a critical analysis of prior work comparing the strengths and weaknesses while also motivating the proposed approach. You are also provided a sentence plan mentioning the total number of lines and the citations to refer in different lines. You should cite all the other related documents as [#] whenever you are referring it in the related work. Do not cite abstract. Do not include any extra notes or newline characters at the end. Do not copy the abstracts of reference papers directly but compare and contrast to the main work concisely. Do not provide the output in bullet points. Do not provide references at the end. Please cite all the provided reference papers. Please follow the plan when generating sentences, especially the number of lines to generate."
sentence_plan = "1. Introduction sentence\n2. Overview of relevant studies\n3. Detailed discussion on key papers\n4. Summary of related work\n"

@app.post("/generateLiteratureSurvey/", response_model=ResponseData)
async def generate_literature_survey(request_data: RequestData):
    summary, ids = summarize(request_data.abstract, request_data.words, request_data.papers)
    return {"summary": summary,
            "ids": ids
            }

@app.get("/")
async def root():
    return {"status": 1}

def summarize(query, n_words, n_papers) :
   keywords = helper.extract_keywords(query)
   papers = helper.search_papers(keywords, n_papers*2)
   ranked_papers = helper.re_rank_papers(query, papers, n_papers)
   literature_review, ids = helper.generate_related_work(query, ranked_papers, base_prompt, sentence_plan, n_words)
   return literature_review, ids

if __name__ == '__main__':
   print("Program running")
   uvicorn.run(app)
