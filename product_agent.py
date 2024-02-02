from datasets import load_dataset
import os
import re
import datetime
import json
from agent_toolkit import AgentToolkit
from managers import RichContextMemory
from callback_handlers import MemoryCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.embeddings.base import Embeddings
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks import StdOutCallbackHandler
import logging

FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""

class ProductAgent:

    def __init__(self, config_path = "./config.json") -> None:
        
        self.load_config(config_path)
        # self.log_manager = LogManager(self.config)

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    

    def init_agent(self, embedding: Embeddings = OpenAIEmbeddings()):
        self.embeddings = embedding

        # self.vector_store = self.document_manager.build_vector_store(embedding)

        self.llm = ChatOpenAI(model = "gpt-4", temperature=0.5, verbose=True)# self.config['llm_settings']['verbose'])
        self.agent_toolkit = AgentToolkit(self.llm, self.config)

        self.build_prompt_template()

        self.memory = self.load_agent_memory()

        self.build_agent()
    
    def message_agent(self, text: str):
        try:
            # Before running agent actions, extract the serialized context data for GPT-4
            serialized_context = self.memory.get_history_serialized()

            langchain_response = self.agent_chain.invoke(
                {
                    "input": text.strip(),
                    "chat_history": serialized_context,
                },
                config = {
                    "callbacks": CallbackManager([MemoryCallbackHandler(memory = self.memory), StdOutCallbackHandler()])              
                }
            )

            # Extract the 'output' from the Langchain response
            output = langchain_response.get('output')if type(langchain_response) is not str else langchain_response
                
            # Store the final interaction (user input and agent's response)
            self.memory.remember(user_input=text, agent_output=output)

            # Return the extracted 'output' from the response
            return output

        except Exception as e:
            # self.log_manager.log_error(f"Error processing the agent response: {e}")
            raise e 

    # def build_prompt_template(self):
    #     doc_list = "" # "\n".join([f'{x.name}: {x.description}' for x in self.document_manager.doc_list])
    #     prefix = (f"You are a helpful personal home assistant for {self.config['prompt_settings']['user_name']}. Your job is to follow his requests and answer his questions to the best of your ability. If you struggle to answer a qustion try thinking about how you could do it step by step, otherwise you can ask the user for help."
    #               "\nToday is " + datetime.datetime.now().strftime(self.config['prompt_settings']['date_format']) + f". Your current location is in {self.config['prompt_settings']['location']}."
    #               # "\n\nYou have access to documents containing the following information:\n\n" + doc_list +
    #               "\n\nYou have access to the following tools:")


    #     suffix = ("Make sure to think \"Start\""
    #              "\n{chat_history}"
    #              "\nQuestion: {input}"
    #             "\n{agent_scratchpad}")
        
    #     self.agent_prompt = ZeroShotAgent.create_prompt(
    #                                 tools = self.agent_toolkit.tool_list, 
    #                                 prefix = prefix, 
    #                                 suffix = suffix, 
    #                                 input_variables = ["input", "chat_history", "agent_scratchpad"]
    #                                 )
        
    def build_prompt_template(self) -> PromptTemplate:
        doc_list = "" # "\n".join([f'{x.name}: {x.description}' for x in self.document_manager.doc_list])
        template = (f"You are a friendly and helpful Home Depot shopping assistant. Your job is to answer consumer questions and help them find what they need. Make sure you are friendly, energetic, and ready to help! If you struggle to answer a qustion try thinking about how you could do it step by step, otherwise you can ask the user for help."
                f"\nYou will be helping {self.config['prompt_settings']['user_name']} today."
                "\nToday is " + datetime.datetime.now().strftime(self.config['prompt_settings']['date_format']) + f". Your current location is in {self.config['prompt_settings']['location']}."
                "\n\nYou have access to documents containing the following information:\n\n" #+ doc_list +
                "Home Depot Product Knowledge Base (CSV file)"
                "\n\nYou have access to the following tools:\n\n{tools}"
                "\n\nUse the following format:"
                "\nQuestion: the input question you must answer"
                "\nThought: you should always think about what to do"
                "\nAction: the action to take, should be one of [{tool_names}]"
                "\nAction Input: the input to the action"
                "\nObservation: the result of the action"
                "\n... (this Thought/Action/Action Input/Observation can repeat N times)"
                "\nThought: I now know the final answer"
                "\nFinal Answer: the final answer to the original input question"
                "\n\nMake sure to think \"Start\""
                "\n\nPrevious chat history:\n{chat_history}"
                "\n\nQuestion: {input}"
                "\n{agent_scratchpad}")

        self.agent_prompt = PromptTemplate(
            template=template,
            input_variables=["input", "tools", "tool_names", "agent_scratchpad", "chat_history"]
        )
        
    def load_agent_memory(self):
        #return ConversationBufferMemory(memory_key = "chat_history", return_messages=True, json_safe=True)
        return RichContextMemory(memory_key="chat_history", return_messages=True, json_safe=True)

    def build_agent(self):
        callback_manager = CallbackManager([StdOutCallbackHandler()]) # MemoryCallbackHandler(memory = self.memory)
        agent = create_react_agent(llm=self.llm, tools=self.agent_toolkit.tool_list, prompt=self.agent_prompt)
        self.agent_chain = AgentExecutor.from_agent_and_tools(
                                        agent = agent, 
                                        tools = self.agent_toolkit.tool_list, 
                                        memory = self.memory, 
                                        verbose = True, # self.config['agent_settings']['verbose'],
                                        return_intermediate_outputs = True,
                                        callbacks = callback_manager
                                        )

    def format_phone_number(self, matchobj):
        sections = matchobj.group(0).split('-')
        return ' - '.join([' '.join(list(section)) for section in sections])

def main():
    assistant = ProductAgent(os.path.dirname(os.path.abspath(__file__)) + "/config.json")
    assistant.init_agent()

    while True:
        try:
            question = input("Question: ")
            if question == "exit":
                break
            response = assistant.message_agent(question)
            print(response)
        except KeyboardInterrupt:
            break
        except TypeError as e:
            print("An error occured: \n")
            print(e)
            break
        except Exception as e:
            print("An error occured: \n")
            print(e)
            break

if __name__ == "__main__":
    main()