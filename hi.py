import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformers import pipeline
from langchain import PromptTemplate, OpenAI, LLMChain
from dotenv import load_dotenv
import requests
import streamlit as st




load_dotenv()

HUGGINGFACEHUB_API_TOKEN=os.getenv("HUGGINGFACE_API")

# Load environment variables from the .env fileprinyttt
#load_dotenv("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")
# load_dotenv(find_dotenv())


# # Access the HUGGINGFACE_API environment variable
# huggingface_api_key = "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";

# # Check if the API key is loaded
# if huggingface_api_key:
#     print("Hugging Face API Key:", huggingface_api_key)
# else:
#     print("Hugging Face API Key not found in the .env file")


st.write("Image to Story App")


# Image to Text Captioning




def main ():

    file=st.file_uploader("Choose an image..",type="jpg")

    if file is not None:
        data= file.getvalue()
        with open(file.name, 'wb') as file:
            file.write(data)
    st.image(file,caption="Uploaded Image",use_column_width=True) 
    scenario1=imgtext(file.name)
    story1=generatestory(scenario1)
    text2speech(story1)


    with st.expander("scenario"):
        st.write(scenario1)

    with st.expander("story"):
        st.write(story1)

    st.audio("audio.flac")
      


if __name__=='__main__':
    main()





def imgtext(image_name):
    try:
        captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        res = captioner(image_name)[0]["generated_text"]  # Use the provided image_name parameter
        print("Image Caption:", res)
        return res
    except Exception as e:
        print("An error occurred:", str(e))
        return None


# def imgtext():
#     try:
#         captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
#         image_url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"
#         res = captioner(image_url)[0]["generated_text"]
#         print("Image Caption:", res)
#         return res
#     except Exception as e:
#         print("An error occurred:", str(e))
#         return None

# Call the imgtext function
# scenario = imgtext()


#   llm


def generatestory(scenario):
    template= """
    You are a story teller so generate a short story , no more than 20 words;
    CONTEXT:{scenario}
    STORY:
    """
    

    prompt=PromptTemplate(template=template, input_variables=["scenario"])
    story_llm = LLMChain(llm=OpenAI( model_name="gpt-3.5-turbo",temperature=1),prompt=prompt, verbose=True)
    story=story_llm.predict(scenario=scenario)

    print(story)

    return story
    
# story=generatestory(scenario)

# print(story)


# text2speech

def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": "Bearer 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"}

    payload = {
        "inputs": message
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        with open("audio.flac", 'wb') as file:
            file.write(response.content)
    else:
        print("Error:", response.status_code)
        print("Response content:", response.content)

# Usage example
# text2speech(story)









