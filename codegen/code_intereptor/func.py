import os
from typing import Optional
import boto3
import json
import requests
from botocore.config import Config
from botocore.config import Config
from langchain.llms.bedrock import Bedrock
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.agents import Tool, AgentExecutor, AgentOutputParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from langchain.chains import LLMChain
from langchain.schema import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import BaseRetriever, Document
from typing import Any, Dict, List, Optional,Union
from langchain.utilities import SerpAPIWrapper
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.memory import ConversationBufferMemory
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
import re


def get_bedrock_client(
    assumed_role: Optional[str] = None,
    region: Optional[str] = None,
    runtime: Optional[bool] = True,
):
  
    if region is None:
        target_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION"))
    else:
        target_region = region

    print(f"Create new client\n  Using region: {target_region}")
    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}

    profile_name = os.environ.get("AWS_PROFILE")
    if profile_name:
        print(f"  Using profile: {profile_name}")
        session_kwargs["profile_name"] = profile_name

    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    session = boto3.Session(**session_kwargs)

    if assumed_role:
        print(f"  Using role: {assumed_role}", end='')
        sts = session.client("sts")
        response = sts.assume_role(
            RoleArn=str(assumed_role),
            RoleSessionName="langchain-llm-1"
        )
        print(" ... successful!")
        client_kwargs["aws_access_key_id"] = response["Credentials"]["AccessKeyId"]
        client_kwargs["aws_secret_access_key"] = response["Credentials"]["SecretAccessKey"]
        client_kwargs["aws_session_token"] = response["Credentials"]["SessionToken"]
        

    if runtime:
        service_name='bedrock-runtime'
    else:
        service_name='bedrock'

    client_kwargs["aws_access_key_id"] = os.environ.get("AWS_ACCESS_KEY_ID","")
    client_kwargs["aws_secret_access_key"] = os.environ.get("AWS_SECRET_ACCESS_KEY","")
    
    bedrock_client = session.client(
        service_name=service_name,
        config=retry_config,
        **client_kwargs
    )

    print("boto3 Bedrock client successfully created!")
    print(bedrock_client._endpoint)
    return bedrock_client



## for aksk bedrock
def get_bedrock_aksk(secret_name='chatbot_bedrock', region_name = "us-west-2"):
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    # Decrypts secret using the associated KMS key.
    secret = json.loads(get_secret_value_response['SecretString'])
    return secret['BEDROCK_ACCESS_KEY'],secret['BEDROCK_SECRET_KEY']

ACCESS_KEY, SECRET_KEY=get_bedrock_aksk()

#role based initial client#######
os.environ["AWS_DEFAULT_REGION"] = "us-west-2"  # E.g. "us-east-1"
os.environ["AWS_PROFILE"] = "default"
#os.environ["BEDROCK_ASSUME_ROLE"] = "arn:aws:iam::687912291502:role/service-role/AmazonSageMaker-ExecutionRole-20211013T113123"  # E.g. "arn:aws:..."
os.environ["AWS_ACCESS_KEY_ID"]=ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"]=SECRET_KEY


#新boto3 sdk只能session方式初始化bedrock
boto3_bedrock = get_bedrock_client(
    #assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None)
)

parameters_bedrock = {
    "max_tokens_to_sample": 2048,
    #"temperature": 0.5,
    "temperature": 0,
    #"top_k": 250,
    #"top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}

bedrock_llm = Bedrock(model_id="anthropic.claude-v2", client=boto3_bedrock, model_kwargs=parameters_bedrock)
memory = ConversationBufferWindowMemory(k=2,memory_key="chat_history", input_key='input', output_key="output")
cot_template = """
        You are an experienced AWS CLI expert who can effectively troubleshoot and resolve issues.
        Generate AWS CLI scripts to troubleshoot user's questions, and use runtime tools to execute the scripts with AWS runtime and get result."
        Ensure the script is clear, concise, and includes specific details about the expected user's question . Specify the length and format for any troubleshooting documentation that needs to be provided."
        ### Note: Please only respond if you have extensive experience with the AWS CLI and can confidently troubleshoot and resolve issues.
        user's question is in the following <question> tags:
        <question>
        {input}
        </question>
        """
    
prompt = PromptTemplate(
    input_variables=["input"], template=cot_template
)
bedrock_chain = LLMChain(llm=bedrock_llm, prompt=prompt, verbose=True)


def execution_request(code:str):
    data = {
        "language": "awscli", 
        "code": code
    }

    response = requests.post("https://localhost:2000", json=data)
    result = response.json()
    return result

runtime_tool = Tool(
    name="run the code with AWS runtime",
    func=execution_request,
    description="useful for when you need to run the code with AWS runtime to get final result",
)
codegen_tool = Tool(
    func=bedrock_chain.run,
    name="AWS CLI script code generation",
    description="useful for when you need to generate aws cli scripte code",
)
custom_tool_list = [runtime_tool,codegen_tool]

customerized_instructions="""
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

These are guidance on when to use a tool to solve a task, follow them strictly:
 - first use "search enterprise documents" tool to retreve the document to answer if need
 - then use "search website" tool to search the latest website to answer if need
"""

##use customerized outputparse to fix claude not match 
##langchain's openai ReAct template don't have 
##final answer issue 
class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        #print("cur step's llm_output ==="+llm_output)
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output},
                log=llm_output,
            )
            #raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
