from argparse import ArgumentParser
from instructor import OpenAISchema
from pydantic import Field
import requests,os
import openai,json
from retrying import retry
from dotenv import load_dotenv
import logging

class Search(OpenAISchema):
    """Busca no edital da FUNDEP"""
    query: str = Field(..., description="The query to be searched.")

    def __call__(self, **kwargs):
        params = {
            "index": kwargs["index"],
            "query": self.query,
            "max_docs_to_return": kwargs["max_docs_to_return"],
            "format_response": False,
            "deduplicate": False,
        }

        key = os.getenv("NSX_API_KEY")
        headers = {
            "Authorization": f"APIKey {key}",
        }

        try:
            response = requests.get(
                url=kwargs["nsx_search_endpoint"],
                params=params,
                headers=headers,
            )
        except requests.exceptions.Timeout as te:
            raise te

        if not response.ok:
            if response.status_code == 403:
                raise Exception("Invalid API key.")
            raise Exception(f"Error in NSX: {response.json()['message']}.")

        response = response.json()

        documents = [
            {"paragraphs": response_reranker["paragraphs"][0]}
            for response_reranker in response["response_reranker"]
        ]
        
        text = ""
        for i,doc in enumerate(documents):
            text += f"Document {i+1}: {doc['paragraphs']}\n"
        return text

class Agent():
    def __init__(self,**args):
        self.args = args
        
        # read .env file
        load_dotenv()

        openai.api_key = os.getenv("OPENAI_API_KEY")

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
                '{"created_date":"%(asctime)s.%(msecs)03d", "module":"%(name)s", "level":"%(levelname)s", "message":"%(message)s"}',
                datefmt="%Y-%m-%dT%H:%M:%S"
        )
            
        file_handler = logging.FileHandler(args["logs_file"], mode='a+')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def call_model(self,messages,functions=None):
        response = openai.ChatCompletion.create(
            model=self.args["model"],
            messages= messages,
            temperature=0,
            functions=functions
        )
        return response
    
    def __call__(self, query):
        instruction = open("./sense/instruction.md","r").read()
        messages = [
            {"role":"system","content":instruction},
            {"role":"user","content":f"Question: {query}"},
        ]

        while True:
            response = self.call_model(messages, functions=[Search.openai_schema])
            finish_reason = response["choices"][0]["finish_reason"]
            if finish_reason == "function_call":
                ai_message = dict(response["choices"][0]["message"])
                function_name = response["choices"][0]["message"]["function_call"]["name"]

                messages.append(ai_message)
            
                function_call = eval(function_name).from_response(response)

                if function_call:
                    function_response = function_call(**self.args)
                    messages.append({"role":"function","name":function_name,"content":function_response})
            else:
                ai_message = dict(response["choices"][0]["message"])
                messages.append(ai_message)
                self.logger.info(json.dumps(messages))
                return response["choices"][0]["message"]["content"]
            


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613")
    # parser.add_argument("--openai_api_key", type=str, required=True)
    # parser.add_argument("--nsx_api_key", type=str, required=True)
    parser.add_argument("--nsx_search_endpoint", type=str, default="https://nsx.ai/api/search")
    parser.add_argument("--index", type=str, default="FUNDEP_Paraopeba")
    parser.add_argument("--language", type=str, default="pt")
    parser.add_argument("--max_docs_to_return", type=int, default=10)
    parser.add_argument("--logs_file", type=str, default="logs/logs.jsonl")

    args = parser.parse_args()
    agent = Agent(**vars(args))
    
    while True:
        query = input("Query: ")
        response = agent(query)
        print(response)