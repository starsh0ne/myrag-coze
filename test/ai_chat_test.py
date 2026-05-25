

from langchain_openai import ChatOpenAI


query='C601房间4号工位的是谁'
#query='早上10点张三在办公室吗'
#query='早上10点李四在办公室吗'
context='李四'
template = """

        在固定场景下：所有人都是固定工位，已知信息都是依据对应问题从对应位置采集的数据，请按流程回答问题。
        1.理解问题后，根据已知信息，生成一个简短的句子，描述与问题相关的情境。
        2.根据重新生成的信息和问题情境，直接回答查询中的问题。
        3.如果没有足够信息回答问题，请直接说明不知道。
        示例：
        问题："早上10点李四在办公室吗"
        已知信息："张三"
        代表从李四原本所在的位置采集的信息是张三，信息不匹配，则意味着李四不在，而张三在办公室
        重生成内容：李四不在，而张三在办公室
        回答：李四不在办公室
        
        start：
        问题：{query}
        已知信息：{context}
    
"""
prompt = template.format(query=query, context=context)
#print(prompt)
#prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model="gpt-3.5-turbo",
    base_url='https://xiaoai.plus/v1',   #url
    api_key='sk-xxx') #your open_ai key



answer=llm.invoke(prompt)
answer=answer.content
print(answer)
data = answer.split("回答：")[1].strip()
print(data)
