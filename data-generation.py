
import torch
import warnings
import os
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from colorama import Fore

import json
from typing import List
from pydantic import BaseModel
from litellm import completion
from generated_prompt import prompt_template

# Suppress MPS warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'


class Record(BaseModel):
    question: str
    answer: str

class Response(BaseModel):
    generated: List[Record]

def llm_call(data: str, num_records: int = 5) -> dict:
    stream = completion(
        model="ollama_chat/qwen2.5:14b",
        messages=[
            {
                "role": "user",
                "content": prompt_template(data, num_records)
            }
        ],
        stream=True,
        options={"num_predict": 2000},
        format=Response.model_json_schema()
    )
    data = ""
    for x in stream:
        delta = x['choices'][0]['delta']['content']
        if delta is not None:
            print(Fore.LIGHTBLUE_EX + delta + Fore.RESET, end="")
            data += delta
    return json.loads(data)


if __name__ == "__main__":

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(Fore.GREEN + "Using Metal Performance Shaders (GPU)" + Fore.RESET)
    else:
        device = torch.device("cpu")
        print(Fore.YELLOW + "Using CPU" + Fore.RESET)

    converter = DocumentConverter()
    doc = converter.convert("cases/ACHEAMFOUR_GROUP_LTD_&_3_ORS_VRS_ANOKYE_&_4_ORS_(J4-03-2024)_[2024]_GHASC_58_(4_December_2024).pdf").document
    chunker = HybridChunker()
    chunks = chunker.chunk(dl_doc=doc)

    dataset = {}
    for i, chunk in enumerate(chunks):
        print(Fore.YELLOW + f"Raw Text: \n{chunk.text[:300]}..." + Fore.RESET)
        enriched_text = chunker.contextualize(chunk=chunk)
        print(Fore.LIGHTMAGENTA_EX + f"Contextualizzed Text: \n {enriched_text[:300]}..." + Fore.RESET)

        data = llm_call(
            enriched_text
        )

        dataset[i] = {"generated": data["generated"], "context": enriched_text}

    with open('tm1data.json', 'w') as f:
        json.dump(dataset, f)

#     example_data_chunk = """Apple silicon is a series of system on a chip (SoC) and system in a package (SiP) processors designed by Apple Inc., mainly using the ARM architecture. They are used in nearly all of the company's devices including Mac, iPhone, iPad, Apple TV, Apple Watch, AirPods, AirTag, HomePod, and Apple Vision Pro.

# The first Apple-designed system-on-a-chip was the Apple A4, which was introduced in 2010 with the first-generation iPad and later used in the iPhone 4, fourth generation iPod Touch and second generation Apple TV.

# Apple announced its plan to switch Mac computers from Intel processors to its own chips at WWDC 2020 on June 22, 2020, and began referring to its chips as Apple silicon.[1][2] The first Macs with Apple silicon, built with the Apple M1 chip, were unveiled on November 10, 2020. The Mac lineup completed its transition to Apple chips in June 2023.

# Apple fully controls the integration of Apple silicon in the company's hardware and software products. Johny Srouji, the senior vice president for Apple's hardware technologies, is in charge of the silicon design.[3] Apple is a fabless manufacturer; production of the chips is outsourced to contract foundries including TSMC and Samsung."""
    # llm_call(example_data_chunk)
