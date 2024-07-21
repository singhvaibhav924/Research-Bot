import uvicorn
import helper
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

llms = None
base_prompt = "You will be provided with an abstract of a scientific document and other references papers in triple quotes. Your task is to write the related work section of the document using only the provided abstracts and other references papers. Please write the related work section creating a cohesive storyline by doing a critical analysis of prior work comparing the strengths and weaknesses while also motivating the proposed approach. You are also provided a sentence plan mentioning the total number of lines and the citations to refer in different lines. You should cite all the other related documents as [#] whenever you are referring it in the related work. Do not cite abstract. Do not include any extra notes or newline characters at the end. Do not copy the abstracts of reference papers directly but compare and contrast to the main work concisely. Do not provide the output in bullet points. Do not provide references at the end. Please cite all the provided reference papers. Please follow the plan when generating sentences, especially the number of lines to generate."
sentence_plan = "1. Introduction sentence\n2. Overview of relevant studies\n3. Detailed discussion on key papers\n4. Summary of related work\n"

class RequestData(BaseModel):
    abstract: str
    words: str
    papers: str

class ResponseData(BaseModel):
    summary: str
    ids: List[str]

@app.post("/generateLiteratureSurvey/", response_model=ResponseData)
async def generate_literature_survey(request_data: RequestData):
    summary, ids = summarize(request_data.abstract, request_data.words, request_data.papers, llms)
    return {"summary": summary,
            "ids": ids
            }

@app.get("/")
async def root():
    if llms == None :
      return {"status": 0}
    return {"status": 1}

@app.get("/test")
async def root():
    if llms == None :
      return {"status": 0}
    return {"status": 1}

def summarize(query, n_words, n_papers, llms) :
   keywords = helper.extract_keywords(llms['feature_extractor'], query)
   papers = helper.search_papers(llms['arxiv_agent'], keywords, int(n_papers)*2)
   ranked_papers = helper.re_rank_papers(llms['ranker'], query, papers, int(n_papers))
   literature_review, ids = helper.generate_related_work(llms['summarizer'], llms['summarizer_tokenizer'], query, ranked_papers, base_prompt, sentence_plan, int(n_words))
   return literature_review, ids

print("Program running")
llms = helper.init_pipeline()
print('Model loaded')

if __name__ == '__main__':
   uvicorn.run(app)
