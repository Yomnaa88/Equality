from flask import Flask, request, jsonify
import os
import nltk
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

nltk.download('punkt')

app = Flask(__name__)

# Download model
if not os.path.exists('phi-2.Q4_K_M.gguf'):
    os.system('wget https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf')

# Disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Callback manager setup
callback_manager = CallbackManager([])

# Creating LlamaCpp instance
llm = LlamaCpp(
    model_path="phi-2.Q4_K_M.gguf",
    temperature=0.1,
    n_gpu_layers=0,
    n_batch=1024,
    callback_manager=callback_manager,
    verbose=True,
    n_ctx=2048
)

# Define templates
templates = {
    "work_experience": """Instruction:
    Extract and summarize the work experience mentioned in the CV provided below. Focus solely on the details related to work history, including job titles, companies, and duration.
    Text: {text}
    Question: {question}
    Output:""",
    
    "certification": """Instruction:
    Extract and summarize the certification history mentioned in the CV provided below. Include details such as degrees earned, institutions attended, and graduation years.
    Text: {text}
    Question: {question}
    Output:""",
    
    "contact_info": """Instruction:
    Extract and provide the contact information mentioned in the CV provided below. Include details such as phone number, email address, and any other relevant contact links.
    Text: {text}
    Question: {question}
    Output:""",
    
    "skills": """Instruction:
    Focus solely on extracting the skills mentioned in the text below, excluding any other details or context. Your answer should consist of concise skills.
    Text: {text}
    Question: {question}
    Output:"""
}

@app.route('/generate_text', methods=['POST'])
def generate_text():
    data = request.get_json()
    question = data.get('question')
    text = data.get('text')

    if not question or not text:
        return jsonify({"error": "Both 'question' and 'text' fields are required."}), 400

    if question == "Please summarize the work experience mentioned in the CV.":
        template_key = "work_experience"
    elif question == "Please summarize the certification history mentioned in the CV without repeating the output only once.":
        template_key = "certification"
    elif question == "Please extract the contact information mentioned in the CV once.":
        template_key = "contact_info"
    elif question == "What are the 6 skills? Please provide a concise short answer of the only(skills) mentioned in the text without repeating the answer.":
        template_key = "skills"
    else:
        return jsonify({"error": "Invalid question provided."}), 400

    prompt = PromptTemplate(template=templates[template_key], input_variables=["question", "text"])
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"question": question, "text": text})

    return jsonify({"generated_text": response})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run( port= 8000)
