import requests
import numpy as np
import arxiv
from langchain.utilities import ArxivAPIWrapper
import os
from dotenv import load_dotenv

load_dotenv() 

HF_API_TOKEN = os.environ.get('HF_API_TOKEN')
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

summarizer_model_name = "microsoft/Phi-3-mini-4k-instruct"
feature_extractor_model_name = "ml6team/keyphrase-extraction-kbir-inspec"
ranker_model_name = "sentence-transformers/all-MiniLM-L6-v2"

def hf_api_call(model_name, payload):
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    return response.json()

def extract_keywords(abstract):
    payload = {"inputs": abstract}
    result = hf_api_call(feature_extractor_model_name, payload)
    keyphrases = np.unique([item['word'].strip() for item in result])
    print(keyphrases)
    return keyphrases

def search_papers(keywords, n_papers):
    arxiv_agent = ArxivAPIWrapper(top_k_results=n_papers, doc_content_chars_max=None, load_max_docs=n_papers+3)
    query = " ".join(keywords)
    results = arxiv_agent.get_summaries_as_docs(query)
    return results

def re_rank_papers(query_abstract, papers, n_papers):
    summaries = {paper.page_content: {"Title": paper.metadata['Title']} for paper in papers}
    summ_list = []

    payload = {
        "inputs": {
            "source_sentence": query_abstract,
            "sentences": list(summaries.keys())
        }
    }
    result = hf_api_call(ranker_model_name, payload)

    for i, key in enumerate(summaries.keys()):
        summ_list.append((key, summaries[key]["Title"], result[i]))
        print((key, summaries[key]["Title"], result[i]))
    summ_list = sorted(summ_list, key=lambda x: x[2], reverse=True)
    summaries = {}
    for i in range(n_papers) :
        summaries[summ_list[i][0]] = {
            "Title" : summ_list[i][1],
            "score" : summ_list[i][2]
        }
    return summaries

def format_abstracts_as_references(papers):
    cite_text = ""
    i = 0
    for key in papers.keys() :
        citation = f"{i+1}"
        cite_text = f"{cite_text}[{citation}]: {key}\n"
        i+=1
    return cite_text

def format_authors(authors):
    formatted_authors = []
    for author in authors:
        name_parts = author.name.split()
        last_name = name_parts[-1]
        initials = ''.join([name[0] for name in name_parts[:-1]])
        formatted_authors.append(f"{last_name} {initials}")
    return ', '.join(formatted_authors)

def to_vancouver_style(entry):
    authors = format_authors(entry.authors)
    title = entry.title
    journal = 'arXiv'
    year = entry.published.year
    arxiv_id = entry.get_short_id()
    return f"{authors}. {title}. {journal}. {year}. arXiv:{arxiv_id}"

def generate_refs(papers) :
    client = arxiv.Client()
    results = []
    for key in papers.keys() :
        search = arxiv.Search(
          query = papers[key]["Title"],
          max_results = 1,
          sort_by = arxiv.SortCriterion.Relevance
        )
        results.append(list(client.results(search))[0])
        
    references = [to_vancouver_style(entry) for entry in results]
    ids = [entry.get_short_id() for entry in results]
    i = 0
    refs = "\n\nReferences:\n"
    for reference in references:
        refs = f"{refs}[{i+1}] {reference}\n"
        i+=1
    return refs, ids


def generate_related_work(query_abstract, ranked_papers, base_prompt, sentence_plan, n_words):
    data = f"Abstract: {query_abstract} \n {format_abstracts_as_references(ranked_papers)} \n Plan: {sentence_plan}"
    complete_prompt = f"{base_prompt}\n```{data}```"
    
    payload = {
        "inputs": complete_prompt,
        "parameters": {
            "max_new_tokens": n_words,
            "temperature": 0.01,
            "do_sample": False
        }
    }
    
    result = hf_api_call(summarizer_model_name, payload)
    print(result)
    related_work = result[0]['generated_text']
    refs, ids = generate_refs(ranked_papers)
    related_work += refs
    
    with open("literature review.txt", "w") as f:
        f.write(related_work)
    
    return related_work, ids