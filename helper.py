from transformers import AutoModelForCausalLM, AutoTokenizer, TokenClassificationPipeline, AutoModelForTokenClassification, pipeline
from langchain_community.utilities import ArxivAPIWrapper
from transformers.pipelines import AggregationStrategy
from sentence_transformers import SentenceTransformer
import arxiv
import numpy as np
import torch

summarizer_model_name = "microsoft/Phi-3-mini-4k-instruct"
feature_extractor_model_name = "ml6team/keyphrase-extraction-kbir-inspec"
ranker_model_name = "sentence-transformers/all-MiniLM-L6-v2"

class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, all_outputs):
        results = super().postprocess(
            all_outputs=all_outputs,
            aggregation_strategy=AggregationStrategy.SIMPLE,
        )
        return np.unique([result.get("word").strip() for result in results])

def init_pipeline() :
    summarizer_model = AutoModelForCausalLM.from_pretrained( 
        summarizer_model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_model_name)
    
    feature_extractor_model = KeyphraseExtractionPipeline(model=feature_extractor_model_name)
    
    ranker_model=SentenceTransformer(ranker_model_name)
    
    arxiv_agent = ArxivAPIWrapper(top_k_results = 5, doc_content_chars_max = None, load_max_docs = 10)
    return {
        "summarizer" : summarizer_model,
        "summarizer_tokenizer" : summarizer_tokenizer,
        "feature_extractor" : feature_extractor_model,
        "ranker" : ranker_model,
        "arxiv_agent" : arxiv_agent
    }

def extract_keywords(model, abstract):
    keyphrases = model(abstract)
    print(keyphrases)
    return keyphrases


def search_papers(arxiv_agent, keywords, n_papers):
    query = " ".join(keywords)
    results = arxiv_agent.get_summaries_as_docs(query)
    #print("arxiv ouptut ")
    #print(results)
    return results

def re_rank_papers(model, query_abstract, papers, n_papers):
    summaries = {paper.page_content : {"Title":paper.metadata['Title']} for paper in papers}
    print(summaries)
    target_embeddings = model.encode([query_abstract])
    summaries_embeddings = model.encode(list(summaries.keys()))

    cosine_similarities = -torch.nn.functional.cosine_similarity(torch.from_numpy(target_embeddings), torch.from_numpy(summaries_embeddings))
    cosine_similarities = cosine_similarities.tolist()

    i = 0
    for key in summaries.keys() :
        summaries[key]["score"] = cosine_similarities[i]
        i+=1
    return dict(sorted(summaries.items(), key=lambda x: x[1]["score"], reverse=True))

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

def generate_related_work(model, tokenizer, query_abstract, ranked_papers, base_prompt, sentence_plan, n_words):
    input_text = f"Abstract: {query_abstract}\n"
    i = 1
    for key in ranked_papers.keys():
        input_text += f"{i+1}. {ranked_papers[key]['Title']} - {key}\n"
        i+=1
    
    data = f"Abstract: {query_abstract} \n {format_abstracts_as_references(ranked_papers)} \n Plan: {sentence_plan}"
    complete_prompt = f"{base_prompt}\n```{data}```"
    messages = [ 
    {"role": "system", "content": "You are a helpful AI assistant."}, 
    {"role": "user", "content": complete_prompt}]
    
    pipe = pipeline( 
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    )

    generation_args = { 
    "max_new_tokens": n_words, 
    "return_full_text": False, 
    "temperature": 0.0, 
    "do_sample": False, 
    } 

    output = pipe(messages, **generation_args) 
    print(output)
    related_work = output[0]['generated_text']
    refs, ids = generate_refs(ranked_papers)
    related_work += refs
    f = open("literature review.txt", "w")
    f.write(related_work)
    f.close()
    return related_work, ids