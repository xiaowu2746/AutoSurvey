from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from typing import TypedDict, Literal
from dotenv import load_dotenv

load_dotenv()

class QueryState(TypedDict):
    topic : str
    queries : list[str]
    iteration: int
    feedback: str
    status : str
    problem: str

class QueryList(BaseModel):
    queries: list[str] = Field(description= 'a list of acedemic words for queries')


class QueryProblem(BaseModel):
    problem: str = Field(description = '描述当前检索词的问题')
    status: Literal['retry', 'enough'] = Field(description= '如果你认为这些检索词符合要求返回enough, 反之返回retry')

class QueryRewrite(BaseModel):
    feedback: str = Field(description= '给当前问题的建议')


def queries_generate(state: QueryState):
    iteration = state.get('iteration', 0) + 1
    feedback = state['feedback']
    topic = state['topic']

    system_prompt = f"""
要求，请根据{topic}给我一些用于检索论文的词，英文，整理成一个python列表
以下是相关建议，{feedback}
"""
    print('Query生成中...')
    response = model.with_structured_output(QueryList).invoke(system_prompt)
    print(f'第{iteration}版query为{response.queries}')
    return{
        'iteration': iteration,
        'queries': response.queries
    }

def queries_grade(state: QueryState):
    queries = state['queries']
    topic = state['topic']
    iteration = state['iteration'] 
    if iteration > 3:
        return {'status': 'enough'}

    system_prompt = f"""
    请根据得到的有关{topic}主题的检索词{', '.join(queries)}是否满足要求，指出当前提示词的问题所在
    具体要求为：
    1. 查询数量是否足够(5-8个)
    2. 是否存在重复表达
    3. 是否全是泛词，没有具体方向
    4. 是否体现survey/review类关键词
    5. 是否缺少benchmark/framework/planning这类更具体的检索角度

    如果有一点不符合要求返回retry,否则返回enough

    """
    
    response = model.with_structured_output(QueryProblem).invoke(system_prompt)
    print({response.status})
    print({response.problem})


    return{
        'status': response.status,
        'problem': response.problem
    }

def queries_rewrite_helper(state: QueryState):
    topic = state['topic']
    queries = state['queries']
    problem = state['problem']

    system_prompt = f"""
    我们在围绕{topic}主题寻找综述检索词，目前这是我们生成的检索词{', '.join(queries)}
    这是我们检索词的问题{problem}。请根据相关问题给我们相关改进意见
    """
    print('生成建议中...')
    response = model.with_structured_output(QueryRewrite).invoke(system_prompt)
    print(f'建议如下， {response.feedback}')
    return {
        'feedback': response.feedback
    }


def route_by_grade(state: QueryState):
    return state['status']

model = init_chat_model('deepseek-chat')

workflow = StateGraph(QueryState)
workflow.add_node('QueryGenerate', queries_generate)
workflow.add_node('QueryGrade', queries_grade)
workflow.add_node('QueryRewrite', queries_rewrite_helper)

workflow.add_edge(START, 'QueryGenerate')
workflow.add_edge('QueryGenerate', 'QueryGrade')
workflow.add_conditional_edges('QueryGrade',
                               route_by_grade,
                               {
                                   'retry': 'QueryRewrite',
                                   'enough': END
                               })
workflow.add_edge('QueryRewrite', 'QueryGenerate')
app = workflow.compile()

result = app.invoke({'topic': 'multi-agent systems for scientific literature survey',
                     'iteration': 0,
                     'feedback': '对于第一次生成检索词并没有feedback'})

print(result)
