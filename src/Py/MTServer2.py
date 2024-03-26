from flask import Flask, jsonify, request
from flask_cors import CORS


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def OpenAIKey():
    Key1 = "sk-UrqKfMDLy4bH"
    Key2 = "60Nxft4JT3BlbkFJptKC"
    Key3 = "Dk1iLXTXOAT0gebM"
    FullKey=Key1+Key2+Key3
    return(FullKey)

APIKey=OpenAIKey()
print(APIKey)

app = Flask(__name__)
CORS(app)

@app.route('/Tokenizer', methods=['POST'])
def Tokenizer():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data = request.get_json()
    TokenizerText = data.get('TokenizerText', '')
    print(TokenizerText)
    sequence = TokenizerText
    print(sequence)
    res = tokenizer(sequence)
    print(res)
    tokens = tokenizer.tokenize(sequence)
    print(tokens)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(ids)
    join = ", ".join(tokens)
    print(join)
    return jsonify(join)

# Model = AutoModelForSequenceClassification.from_pretrained("./mt_ml_models/MT_DS_HP1_Mood_Bal_v6_Model/")
# Tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# Moods = ['Neutral', 'Happy', 'Love', 'Excite', 'Sad', 'Anger', 'Fear']

# @app.route('/SentimentAnalysis4', methods=['POST'])
# def SentimentAnalysis4():
#     data = request.get_json()
#     UserResponse = data.get('UserResponse', '')
#     print(UserResponse)

#     InputData = Tokenizer(UserResponse, truncation=True, padding=True, return_tensors="pt")
#     Logits = Model(**InputData).logits
#     PredictedClass = Logits.argmax().item()
#     PredictedMood = Moods[(PredictedClass)-1]

#     print("\n\nPredictedLabel: ", PredictedMood, "\n\n")
#     return jsonify(PredictedMood)


StoryPrompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a mystery story writer"
            "The setting is: The main character is trapped in a room with six characters named Halin, Leo, Ethi, Skott, Ariadni and Frikyn each representing Happy, Love, Excite, Sad, Anger and Fear respectively. A stranger enters the room. All the people in the room has to escape the room by working together. Should they trust each other? Who is the stranger?"
            "Your writing should be around the main character, the other six characters and the stranger, all trying to get out of the locked room."
            "Mood of the scene is {Mood}"
            "Refer to main character as you, refer to other characters with their name, refer to the stranger as The Stranger."
            "In the story do not mention that the other characters represent moods"
            "Your response should end with a specific scenario prompting user to respond"
            "Your output should be 50 words"
        ),
        ("human", "{text}"),
    ]
)

LLM = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=APIKey)

Runnable = StoryPrompt | LLM

@app.route('/TextGeneration4', methods=['POST'])
def GenerateStory():

    data = request.get_json()

    JoinedHistory = data.get('UserResponse', '')

    Mood = data.get('Mood', '')
    print(Mood)
    
    StoryResponse = Runnable.invoke({"text": JoinedHistory, "Mood": Mood})

    StorySegment = StoryResponse.content
    
    print("\n")
    print(StoryResponse)
    print("\n")
    print("\n")
    print(StorySegment)
    print("\n")

    return jsonify(StorySegment)

if __name__ == '__main__':
    app.run(host='localhost')
