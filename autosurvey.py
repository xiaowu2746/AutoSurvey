from typing import TypedDict
from langgraph.graph import StateGraph, END, START

class ResearchState(TypedDict):
    interation: int
    topic: str
    queries: list[str]
    paper_count : int 
    status : str


def queries_generate(state: ResearchState) -> dict:
    topic = state['topic']
    interation = state.get('interation', 0) + 1
    if interation == 1:
        queries = [
            topic,
            f'{topic} survey',
            f'{topic} benchmark'
        ]
    else:
        queries = [
            topic,
            f'{topic} I',
            f'{topic} II',
            f'{topic} III'
        ]

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
