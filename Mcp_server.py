import asyncio
import base64
import json
import os
import torch
import requests

from bs4 import BeautifulSoup
from email.mime.text import MIMEText
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

def loadModel():
    # print("model loading")
    bnb = BitsAndBytesConfig(load_in_4bit=True)

    true_model = AutoModelForCausalLM.from_pretrained(
        "aaditya/Llama3-OpenBioLLM-8B",
        quantization_config=bnb,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained("MagazinePuma/disease-tuned-llama3-openbiollm")
    tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(true_model, "MagazinePuma/disease-tuned-llama3-openbiollm")
    # print("model loaded")

    return model, tokenizer

def get_user_gmail():
    user_info = None

    if os.path.exists("/Users/dennis/Desktop/Research/Tensorflow/token.json"):
        user_info = Credentials.from_authorized_user_file("/Users/dennis/Desktop/Research/Tensorflow/token.json", SCOPES)

    if not user_info or not user_info.valid:
        
        if user_info and user_info.expired and user_info.refresh_token:
            user_info.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("/Users/dennis/Desktop/Research/Tensorflow/credentials.json", SCOPES)
            user_info = flow.run_local_server(port=0)

        with open("/Users/dennis/Desktop/Research/Tensorflow/token.json", "w") as file:
            file.write(user_info.to_json())

    return build("gmail", "v1", credentials=user_info)

app = Server("agent")
#model, tokenizer = loadModel()
# response = "You might have Influenza."
gmail = get_user_gmail()

@app.list_tools()
async def tools() -> list[Tool]:
    return [
        Tool(
            name = "diagnosis", # change/add tool to send medicine info, not diagnosis
            description = "Predicts possible illness from symptoms",
            inputSchema = {
                "type": "object",
                "properties": {
                    "symptoms": {
                        "type": "string",
                        "description": "symptoms' description"
                    }
                },
                "required": ["symptoms"]
            }
        ),
        Tool(
            name = "send_email",
            description = "sends email through Gmail",
            inputSchema = {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "email to send to"},
                    "subject": {"type": "string", "description": "subject"},
                    "body": {"type": "string", "description": "body"}
                },
                "required": ["to", "subject", "body"]
            }
        ),
        Tool(   # finish creating this and add another tool to actually find which medicine to prescribe
            name = "treatment_and_medication",
            description = "prescribes medicine",
            inputSchema = {
                "type": "object",
                "properties": {
                    "disease": {
                        "type": "string",
                        "description": "medicine/treatment for predicted illness"
                    }
                },
                "required": ["disease"]
            }
        )
    ]
 
@app.call_tool()
async def use_tool(name: str, args: dict) -> list[TextContent]:

    if name == "diagnosis":

        # symptoms = args["symptoms"]
        # prompt = f"<|user|>\nI have {symptoms}. What disease might I have?\n<|assistant|>\n"
        # inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
        # 
        # with torch.no_grad():
        #     output = model.generate(**inputs, max_new_tokens = 100)

        # response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("<|assistant|>")[-1].strip()
        response = "You might have Influenza."

        return[TextContent(type="text", text=response)]
    
    elif name == "send_email":
        to = args["to"]
        subject = args["subject"]
        body = args["body"]

        msg = MIMEText(body)
        msg["to"] = to
        msg["subject"] = subject

        code = base64.urlsafe_b64encode(msg.as_bytes()).decode()

        gmail.users().messages().send(userId = "me", body = {"raw": code}).execute()

        return [TextContent(type="text", text=f"sent email of info to {to}")]
    
    elif name == "treatment_and_medication":
        # finish this
        disease = args["disease"]
        # url = f"https://medlineplus.gov/search/?query={disease.replace(' ', '+')}" # url for database's search
        url = f"https://vsearch.nlm.nih.gov/vivisimo/cgi-bin/query-meta?v%3aproject=medlineplus&v%3asources=medlineplus-bundle&query={disease.replace(' ', '+')}" # url for database's search
        headers = {"User-Agent": "Mozilla/5.0"} # required identification/search paramaters for requests
        search = requests.get(url, headers = headers)

        parsed = BeautifulSoup(search.text, "html.parser")
        # result = parsed.find("a", class_ = "result-title") # find html tag

        result = ( # find tag; try all cases
            parsed.find("a", class_="result-title") or
            parsed.find("li", class_="search-result") or
            parsed.find("div", class_="search-result") or
            parsed.find("a", class_="search-result-title") or
            parsed.find("p", class_="result-title") or
            parsed.find("ul", id_="search-results") or
            parsed.find("ol", class_="results")
        )

        # disease_url = f"https://medlineplus.gov" + result["href"]
        result_link = None

        if result:
            first_li = result.find("li")

            if first_li:
                doc_header = first_li.find("div", class_ = "document-header")

                if doc_header:
                    result_link = doc_header.find("a", class_ = "title")

        if not result_link:
            links = parsed.find_all("a", href=True)

            for link in links:  # find first result link, if can't directly find page
                if "/medlineplus/" in link["href"] and "search" not in link["href"]:
                    result = link
                    break

        if not result:
            return [TextContent(type="text", text=f"No results found for {disease} on MedlinePlus Database.")]
        
        # disease_url = f"https://medlineplus.gov" + result["href"]
        # disease_url = result_link["href"]

        if result_link:
            href = result_link["href"]
            
            if "url=" in href:
                from urllib.parse import urlparse, parse_qs, unquote # library to parse the link easier
                actual_url = unquote(href.split("url=")[1].split("&")[0])
                disease_url = actual_url

            elif href.startswith("http"):
                disease_url = href

            else:
                disease_url = "https://medlineplus.gov" + href

        # response = requests.get(disease_url, headers = headers)
        # parsed_page = BeautifulSoup(response.text, "html.parser")

        # description =(  # find information on html page
        #     parsed_page.find("div", id = "topic-summary") or
        #     parsed_page.find("div", class_ = "section-body") or
        #     parsed_page.find("div", id = "content") or
        #     parsed_page.find("main") or
        #     parsed_page.find("article")# or
        #     # parsed_page.find("section", id_ = "topsum_section")
        # )

        # if not description:
        #     content = parsed_page.find("div", class_ = "section-body") # find actual description tags

        if disease_url: # description:
            # text = description.get_text(separator = " ", strip = True)
            text = disease_url
        else:
            text = "no information on this illness"

        return [TextContent(type="text", text=f"care information for {disease}:\n{text}")]



async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
