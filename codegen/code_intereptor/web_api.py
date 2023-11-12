from fastapi import FastAPI,Request
import uvicorn
import asyncio
import func



app = FastAPI()


def gen_and_execute(query str):
    output_parser = CustomOutputParser()
    agent_executor = initialize_agent(custom_tool_list, bedrock_llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
                                  verbose=True,max_iterations=3,
                                  handle_parsing_errors=True,
                                  memory=memory,
                                  agent_kwargs={
                                      "output_parser": output_parser,
                                      #'prefix':PREFIX,
                                      #'suffix':SUFFIX,
                                      'format_instructions':customerized_instructions
                                           })
    agent_executor.run(query)
    

@app.get("/api/gen_and_execute")
async def call_function(request:Request):
    result = await gen_and_execute(request.query_params['query']) 
    return {"result":result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)