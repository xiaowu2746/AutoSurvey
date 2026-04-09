from typing import TypedDict
from langgraph.graph import StateGraph, END, START
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv(override= True)

class ResearchState(TypedDict):
    interation: int
    topic: str
    queries: list[str]
    paper_count : int 
    status : str


def queries_generate(state: ResearchState) -> dict:
    topic = state['topic']
    interation = state.get('interation', 0) + 1
    system_prompt = f""""
    请你根据主题要求给出适合论文检索的queries，满足一下要求 
    1.给我一个python列表，里面装着每个queries的英文 
    2.如果Interation为1那么请给我三个queries,如果interation大于1，给我5个queries 
    3.不要解释 只给出列表即可

    topic: {state['topic']}
    interation: {state['interation']}
    
    """
    response = model.invoke(system_prompt)
    queries = response.content.strip()

    try:
        queries = eval(queries)
        if not isinstance(queries, list):
            queries = [queries]

    except Exception as e:
        queries = [queries]


    return {'queries': queries,
            'interation': interation}

def search_paper(state: ResearchState) -> dict:
    queries = state['queries']
    count = len(queries)
    return {'paper_count': 3*count}

def evaluate_paper(state: ResearchState) -> dict:
    count = state['paper_count']
    interation = state['interation']

    if interation< 2  and count < 10:
        return {'status': 'retry'}
    
    else:
        return {'status': 'enough'}
    
def rounte_by_evaluate(state: ResearchState):
    return state['status']


model = init_chat_model('deepseek-chat')


workflow = StateGraph(ResearchState)
workflow.add_node('Queries', queries_generate)
workflow.add_node('PaperSearch', search_paper)
workflow.add_node('Evaluation', evaluate_paper)
workflow.add_edge(START, 'Queries')
workflow.add_edge('Queries', 'PaperSearch')
workflow.add_edge('PaperSearch', 'Evaluation')
workflow.add_conditional_edges(
    'Evaluation',
    rounte_by_evaluate,
    {
        'retry': 'Queries',
        'enough': END
    }
)

app = workflow.compile()
result = app.invoke({'topic': 'rag',
                     'interation': 0})

print(result)
