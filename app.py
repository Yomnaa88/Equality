from flask import Flask, request, jsonify
import os
os.system('wget https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf')
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.callbacks.tracers import ConsoleCallbackHandler
import matplotlib.pyplot as plt
import numpy as np
import nltk
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

import nltk
nltk.download('punkt')

app = Flask(__name__)

# Define the text variable with corrected formatting
cv_text  = '''{
    "work experience": " hfyyskl/lkf;os mjdioslld mkidido.d kdodlf",
    "certification history": "hggdyud",
    "skills ": "nshskdllc"
}'''

# Disable GPU usage
# Disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Callback manager setup
callback_manager = CallbackManager([])

# Creating LlamaCpp instance
llm = LlamaCpp(
    model_path="phi-2.Q4_K_M.gguf",  # Using environment variable for model path
    temperature=0.1,
    n_gpu_layers=0,
    n_batch=1024,
    callback_manager=callback_manager,
    verbose=True,
    n_ctx=2048
)


template_certification = """Instruction:
Extract and summarize the certification history mentioned in the CV provided below. Include details such as degrees earned, institutions attended, and graduation years.
Text: {text}
Question: {question}
Output:"""

template_work_experience = """Instruction:
Extract and summarize the work experience mentioned in the CV provided below. Focus solely on the details related to work history, including job titles, companies, and duration.
Text: {text}
Question: {question}
Output:"""

template_contact_info = """Instruction:
Extract and provide the contact information mentioned in the CV provided below. Include details such as phone number, email address, and any other relevant contact links.
Text: {text}
Question: {question}
Output:"""

template_skills = """Instruction:
Focus solely on extracting the skills mentioned in the text below, excluding any other details or context. Your answer should consist of concise skills.
Text: {text}
Question: {question}
Output:"""

@app.route('/generate_text', methods=['POST'])
def generate_text():
    data = request.get_json()
    question = data.get('question')
    text = data.get('text')

    if question == "Please summarize the work experience mentioned in the CV.":
        # Creating PromptTemplate instance for work experience
        prompt_work_experience = PromptTemplate(template=template_work_experience, input_variables=["question","text"])
        chain_work_experience = prompt_work_experience | llm | StrOutputParser()

        # Invoke the work experience chain
        ans_work_experience = chain_work_experience.invoke({"question": question, "text": text})

        return jsonify({"generated_text": ans_work_experience})

    elif question == "Please summarize the certification history mentioned in the CV wihout repeating the output only once.":
        # Creating PromptTemplate instance for certification
        prompt_certification = PromptTemplate(template=template_certification, input_variables=["question","text"])
        chain_certification = prompt_certification | llm | StrOutputParser()

        # Invoke the certification chain
        ans_certification = chain_certification.invoke({"question": question, "text": text})

        return jsonify({"generated_text": ans_certification})

    elif question == "Please extract the contact information mentioned in the CV once.":
        # Creating PromptTemplate instance for contact information
        prompt_contact_info = PromptTemplate(template=template_contact_info, input_variables=["question","text"])
        chain_contact_info = prompt_contact_info | llm | StrOutputParser()

        # Invoke the contact information chain
        ans_contact_info = chain_contact_info.invoke({"question": question, "text": text})

        return jsonify({"generated_text": ans_contact_info})

    elif question == "What are the 6 skills? Please provide a concise short answer of the only(skills) mentioned in the text without repeating the answer.":
        # Define the PromptTemplate instance for skills
        prompt_skills = PromptTemplate(template=template_skills, input_variables=["question","text"])
        # Create a chain for extracting skills
        chain_skills = prompt_skills | llm | StrOutputParser()

        # Invoke the chain to extract skills
        ans_skills = chain_skills.invoke({"question": question, "text": text})

        return jsonify({"generated_text": ans_skills})

    else:
        return jsonify({"error": "Invalid question provided."})



# Sample texts
texts = [
    "JOB1: AWS Certified SysOps Administrator, Java,React.js,MySQL,Git,AWS,Big O notation,Agile,Docker, "
    "computer science, and artificial intelligence.",
    "JOB2: IT Business Analyst,Confluence,Power BI,Microsoft Visio,Rally,iRise,Postman,MicroStrategy,Accompa.",
    "JOB3: Junior Business Analyst,LibreOffice Calc; QlikView; Jira; Salesforce; SAP BusinessObjects; IBM SPSS.",
    "JOB4: Business Intelligence Engineer,Amazon Redshift,Apache NiFi,Tableau,Python,Microsoft SQL Server,Apache Spark,Google Cloud Platform (GCP),Scikit-learn,Power BI,Talend.",
    "JOB5: Web Developer,JavaScript,Visual Studio Code,Git,TensorFlow,Selenium (automated testing),Docke."
    "JOB6: Software Engineer,JavaScript (Angular),HTML/ CSS,Python (Django),SQL (PostgreSQL, Oracle),REST APIs (GraphQL),Git, "
    "computer science, and artificial intelligence.",
    "JOB7: Computer TechnicianMicrosoft Windows 10,Memtest86,TeamViewer,McAfee Endpoint Security,Acronis True Image,Recuva,Wireshark,HWiNFO,VMware Workstation,PowerShell.",
    "JOB8: Senior Computer Vision ScientistSnowflake,Python,C++,R,SQL,Tableau,PyTorch.",
    "JOB9:Entry-Level Cyber Security,LogRhythm; Cisco Firepower; Rapid7 Nexpose; Cisco Stealthwatch; IDA Pro; Netsparker; Fortinet FortiGate;CrowdStrike Falcon; KeePass; NordVPN.",
    "JOB10: Cyber Security Intern,pfSense; Snort; Bitdefender; OpenVPN; Splunk; Nessus."
    "JOB11: Cyber Security Engineer,Cisco ASA,Snort,Symantec Endpoint Protection,OpenVPN,Splunk,Microsoft BitLocker,ModSecurity,Nmap.",
    "JOB12: Alteryx Data Analys,Workflow canvas; Data preparation tools; Predictive modeling tools; Data visualization; Interactive dashboards; Database querying; Data manipulation; Data extraction.",
    "JOB13: Data Quality AnalystInformatica Data Quality; OpenRefine; Talend Data Quality; TIBCO EBX; Collibra; Ataccama ONE; QuerySurge; Alation; Sisense; Apache NiFi.",
    "JOB14:SQL Data Analyst,Tableau Desktop, Tableau Server,SQL (Postgres, Redshift, MySQL),PL/SQL,Triggers, stored procedures, views,MS Excel.",
    "JOB15: Senior Data Analyst,Programming: Python (Scikit-learn),SQL,Data Visualization: Tableau, Excel,,Google Sheets, Matplotlib,Modeling: Logistic regressions, linear,regressions, decision trees,Product Analytics: Google Analytics,,A/B Testing & Experimentation."
    "JOB16: Senior Business Intelligence DataEngineer,· DB2,SQL,Snowflake,Python,Java,WebFOCUS,Tableau,MicroStrategy,MongoDB,MySQL.",
    "JOB17: Data Engineer,Python; Java; Power BI; SQL; Redshift; Postgres; Snowflake; AWS; Microsoft SSIS.",
    "JOB18: Data Scientist Intern,NumPy,Scikit-learn,dplyr,MySQL,SQLite.",
    "JOB19:Entry-Level Data Scientist,Programming: SAS (baseSAS and Macros), SQL,Supervised Learning:,linear and logistic,regressions, decision,trees, support vector,machines (SVM),Unsupervised Learning: k means clustering,principal component,analysis (PCA).",
    "JOB20: Python Data Scientist,Python; Tableau; Pandas; PyTorch; Hadoop; spaCy; AWS; MySQL."
    "JOB21:Front-End DeveloperJavaScriptHTML,CSS,React.",
    "JOB22: React Developer,React Router; ESLint; Chakra UI; Redux",
    "JOB23:Senior Front-End Developer,HTML; CSS; JavaScript; React; jQuery; Angular.js",
    "JOB23:UI Front-End Developer,HTML; CSS; JavaScript; Angular.js",

    "JOB24:Full-Stack Software Developer,JavaScript,HTML,CSS,React.js,Node.js,Angular.js,MongoDB.",
    "JOB25: Senior Full Stack Developer,JavaScript; CircleCI; BitBucket; TravisCI; Python; Angular.js; Vue.js; React.js; Node.js; HTML;,CSS.",
    "JOB126:Technical Support,ServiceNow; Remote Desktop Protocol; PRTG Network Monitor; Symantec Endpoint Protection; Linux; Chef;,Microsoft PowerPoint;.",
    "JOB26:Java Developer,Java; JavaScript; Angular.js; HTML; CSS; UNIX; SQL; Eclipse; Oracle; React.js.",
    "JOB27:Entry Level Java Developer,Java,HTML,CSS,JavaScript,C/C++",
    "JOB28:Machine Learning Intern,Microsoft Office; Google Workspace; Jupyter; Matplotlib; Pandas; TensorFlow."   "Text 19:Entry-Level Data Scientist,Programming: SAS (baseSAS and Macros), SQL,Supervised Learning:,linear and logistic,regressions, decision,trees, support vector,machines (SVM),Unsupervised Learning: k means clustering,principal component,analysis (PCA).",
    "JOB29:Cloud Network Engineer,AWS; VLANs; Fortinet FortiGate; VMware vSphere; DockerL.",
    "JOB30:Senior Network Engineer,Cisco - CCNA Certification; LAN/ WAN, TCP/IP Networking; Cisco NEXUS /ISE / Prime (WiFi),CCNA;,FortiManager/ FortiGate; Amazon EC2/ Direct Connection",
    "JOB31:Senior Program Manager,Task Management; Organization; Project Ownership; Excel, Project; Agile methodologies",
    "JOB32:Entry-Level Programmer,AWS; JavaScript; HTML; CSS; React.js/Redux; Angular.js; Node.js; Python",
    "JOB33:Entry-Level Programmer,AWS; JavaScript; HTML; CSS; React.js/Redux; Angular.js; Node.js; Python",
    "JOB34:Senior Programmer,Languages: Python, Javascript,,HTML5/CSS,Frameworks: Django, NodeJS, ReactJS,Tools: jQuery, Unix",

    "JOB35:Quality Assurance,TestRail; Appium; Redmine; Jenkins",
    "TJOB36: Quality Analyst,TestLink,Bugzilla,Appium,LoadRunner.",
    "JOB37:Quality Assurance Manager,JUnit,Bugzilla,Locust,Perforce.",
    "JOB38:UI/UX Designer,Adobe XD,Optimal Workshop,Jira,Adobe Photoshop,HTML/CSS,Sketch,UserTesti.",
    "JOB39:Freelance Web Developer,Bootstrap,React,Git,Visual Studio Code,WordPress,MySQL.",
    "JOB40:Quality Control Specialist,Calipers; Pareto Chart; 5 Whys Technique; Microsoft Office.",
    "JOB41: Quality Control Manager.ISO 9001:2015 Standard,Microsoft Excel,Fishbone (Ishikawa) Diagram,Value Stream Mapping,AuditBoard,Microsoft PowerPoin.",
    "JOB42:AWS Certified Solutions Architect,AWS,OpenShift,Chef,KVM,Palo Alto Networks,Nginx,PostgreSQe.",
]

# Tokenize texts
tokenized_texts = [nltk.word_tokenize(text.lower()) for text in texts]

# Train or load pre-trained word embeddings
# Here, we're using Word2Vec to train embeddings on the tokenized texts
word_embeddings_model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)

# Function to calculate text embeddings by averaging word embeddings
def text_embedding(text):
    words = nltk.word_tokenize(text.lower())
    embeddings = []
    for word in words:
        if word in word_embeddings_model.wv:
            embeddings.append(word_embeddings_model.wv[word])
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(word_embeddings_model.vector_size)  # Return zero vector if no words found

# Calculate embeddings for all texts
text_embeddings = [text_embedding(text) for text in texts]

# Specify the input text
input_text = "UI Front-End Developer,HTML; CSS; JavaScript; Angular.js"

# Calculate cosine similarity
input_embedding = text_embedding(input_text)
similarities = cosine_similarity([input_embedding], text_embeddings).flatten()

# Convert similarity scores to percentages
similarities_percentages = [similarity * 100 for similarity in similarities]

# Display similarity percentages
for i, percentage in enumerate(similarities_percentages):
    print(f"Similarity with Text {i+1}: {percentage:.2f}%")


# Plotting
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size if needed
texts_for_plotting = [text.split(":")[0] for text in texts[1:]]  # Extracting text labels
ax.bar(texts_for_plotting, similarities_percentages[1:])  # excluding input text
ax.set_ylabel('Similarity (%)')
ax.set_xlabel('Texts')
ax.set_title('Similarity of Input Text with other texts')
plt.xticks(rotation=45, ha='right')  # Rotate and align the x-axis labels for better readability
plt.tight_layout()
plt.show()


# Sample texts
texts = [
    "JOB1: AWS Certified SysOps Administrator, Java,React.js,MySQL,Git,AWS,Big O notation,Agile,Docker, "
    "computer science, and artificial intelligence.",
    "JOB2: IT Business Analyst,Confluence,Power BI,Microsoft Visio,Rally,iRise,Postman,MicroStrategy,Accompa.",
    "JOB3: Junior Business Analyst,LibreOffice Calc; QlikView; Jira; Salesforce; SAP BusinessObjects; IBM SPSS.",
    "JOB4: Business Intelligence Engineer,Amazon Redshift,Apache NiFi,Tableau,Python,Microsoft SQL Server,Apache Spark,Google Cloud Platform (GCP),Scikit-learn,Power BI,Talend.",
    "JOB5: Web Developer,JavaScript,Visual Studio Code,Git,TensorFlow,Selenium (automated testing),Docke."
    "JOB6: Software Engineer,JavaScript (Angular),HTML/ CSS,Python (Django),SQL (PostgreSQL, Oracle),REST APIs (GraphQL),Git, "
    "computer science, and artificial intelligence.",
    "JOB7: Computer TechnicianMicrosoft Windows 10,Memtest86,TeamViewer,McAfee Endpoint Security,Acronis True Image,Recuva,Wireshark,HWiNFO,VMware Workstation,PowerShell.",
    "JOB8: Senior Computer Vision ScientistSnowflake,Python,C++,R,SQL,Tableau,PyTorch.",
    "JOB9:Entry-Level Cyber Security,LogRhythm; Cisco Firepower; Rapid7 Nexpose; Cisco Stealthwatch; IDA Pro; Netsparker; Fortinet FortiGate;CrowdStrike Falcon; KeePass; NordVPN.",
    "JOB10: Cyber Security Intern,pfSense; Snort; Bitdefender; OpenVPN; Splunk; Nessus."
    "JOB11: Cyber Security Engineer,Cisco ASA,Snort,Symantec Endpoint Protection,OpenVPN,Splunk,Microsoft BitLocker,ModSecurity,Nmap.",
    "JOB12: Alteryx Data Analys,Workflow canvas; Data preparation tools; Predictive modeling tools; Data visualization; Interactive dashboards; Database querying; Data manipulation; Data extraction.",
    "JOB13: Data Quality AnalystInformatica Data Quality; OpenRefine; Talend Data Quality; TIBCO EBX; Collibra; Ataccama ONE; QuerySurge; Alation; Sisense; Apache NiFi.",
    "JOB14:SQL Data Analyst,Tableau Desktop, Tableau Server,SQL (Postgres, Redshift, MySQL),PL/SQL,Triggers, stored procedures, views,MS Excel.",
    "JOB15: Senior Data Analyst,Programming: Python (Scikit-learn),SQL,Data Visualization: Tableau, Excel,,Google Sheets, Matplotlib,Modeling: Logistic regressions, linear,regressions, decision trees,Product Analytics: Google Analytics,,A/B Testing & Experimentation."
    "JOB16: Senior Business Intelligence DataEngineer,· DB2,SQL,Snowflake,Python,Java,WebFOCUS,Tableau,MicroStrategy,MongoDB,MySQL.",
    "JOB17: Data Engineer,Python; Java; Power BI; SQL; Redshift; Postgres; Snowflake; AWS; Microsoft SSIS.",
    "JOB18: Data Scientist Intern,NumPy,Scikit-learn,dplyr,MySQL,SQLite.",
    "JOB19:Entry-Level Data Scientist,Programming: SAS (baseSAS and Macros), SQL,Supervised Learning:,linear and logistic,regressions, decision,trees, support vector,machines (SVM),Unsupervised Learning: k means clustering,principal component,analysis (PCA).",
    "JOB20: Python Data Scientist,Python; Tableau; Pandas; PyTorch; Hadoop; spaCy; AWS; MySQL."
    "JOB21:Front-End DeveloperJavaScriptHTML,CSS,React.",
    "JOB22: React Developer,React Router; ESLint; Chakra UI; Redux",
    "JOB23:Senior Front-End Developer,HTML; CSS; JavaScript; React; jQuery; Angular.js",
    "JOB23:UI Front-End Developer,HTML; CSS; JavaScript; Angular.js",

    "JOB24:Full-Stack Software Developer,JavaScript,HTML,CSS,React.js,Node.js,Angular.js,MongoDB.",
    "JOB25: Senior Full Stack Developer,JavaScript; CircleCI; BitBucket; TravisCI; Python; Angular.js; Vue.js; React.js; Node.js; HTML;,CSS.",
    "JOB126:Technical Support,ServiceNow; Remote Desktop Protocol; PRTG Network Monitor; Symantec Endpoint Protection; Linux; Chef;,Microsoft PowerPoint;.",
    "JOB26:Java Developer,Java; JavaScript; Angular.js; HTML; CSS; UNIX; SQL; Eclipse; Oracle; React.js.",
    "JOB27:Entry Level Java Developer,Java,HTML,CSS,JavaScript,C/C++",
    "JOB28:Machine Learning Intern,Microsoft Office; Google Workspace; Jupyter; Matplotlib; Pandas; TensorFlow."   "Text 19:Entry-Level Data Scientist,Programming: SAS (baseSAS and Macros), SQL,Supervised Learning:,linear and logistic,regressions, decision,trees, support vector,machines (SVM),Unsupervised Learning: k means clustering,principal component,analysis (PCA).",
    "JOB29:Cloud Network Engineer,AWS; VLANs; Fortinet FortiGate; VMware vSphere; DockerL.",
    "JOB30:Senior Network Engineer,Cisco - CCNA Certification; LAN/ WAN, TCP/IP Networking; Cisco NEXUS /ISE / Prime (WiFi),CCNA;,FortiManager/ FortiGate; Amazon EC2/ Direct Connection",
    "JOB31:Senior Program Manager,Task Management; Organization; Project Ownership; Excel, Project; Agile methodologies",
    "JOB32:Entry-Level Programmer,AWS; JavaScript; HTML; CSS; React.js/Redux; Angular.js; Node.js; Python",
    "JOB33:Entry-Level Programmer,AWS; JavaScript; HTML; CSS; React.js/Redux; Angular.js; Node.js; Python",
    "JOB34:Senior Programmer,Languages: Python, Javascript,,HTML5/CSS,Frameworks: Django, NodeJS, ReactJS,Tools: jQuery, Unix",

    "JOB35:Quality Assurance,TestRail; Appium; Redmine; Jenkins",
    "TJOB36: Quality Analyst,TestLink,Bugzilla,Appium,LoadRunner.",
    "JOB37:Quality Assurance Manager,JUnit,Bugzilla,Locust,Perforce.",
    "JOB38:UI/UX Designer,Adobe XD,Optimal Workshop,Jira,Adobe Photoshop,HTML/CSS,Sketch,UserTesti.",
    "JOB39:Freelance Web Developer,Bootstrap,React,Git,Visual Studio Code,WordPress,MySQL.",
    "JOB40:Quality Control Specialist,Calipers; Pareto Chart; 5 Whys Technique; Microsoft Office.",
    "JOB41: Quality Control Manager.ISO 9001:2015 Standard,Microsoft Excel,Fishbone (Ishikawa) Diagram,Value Stream Mapping,AuditBoard,Microsoft PowerPoin.",
    "JOB42:AWS Certified Solutions Architect,AWS,OpenShift,Chef,KVM,Palo Alto Networks,Nginx,PostgreSQe.",
]

# Tokenize texts
tokenized_texts = [nltk.word_tokenize(text.lower()) for text in texts]

# Train or load pre-trained word embeddings
# Here, we're using Word2Vec to train embeddings on the tokenized texts
word_embeddings_model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)

# Function to calculate text embeddings by averaging word embeddings
def text_embedding(text):
    words = nltk.word_tokenize(text.lower())
    embeddings = []
    for word in words:
        if word in word_embeddings_model.wv:
            embeddings.append(word_embeddings_model.wv[word])
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(word_embeddings_model.vector_size)  # Return zero vector if no words found

# Calculate embeddings for all texts
text_embeddings = [text_embedding(text) for text in texts]

# Specify the input text
input_text ="UI Front-End Developer,HTML; CSS; JavaScript; Angular.js"
# Calculate cosine similarity
input_embedding = text_embedding(input_text)
similarities = cosine_similarity([input_embedding], text_embeddings).flatten()

# Get indices of texts sorted by similarity
sorted_indices = np.argsort(similarities)[::-1]

similar_texts = []

# Display the three most similar texts and their similarity percentages
for i in range(3):
    index = sorted_indices[i]
    similarity_percentage = similarities[index] * 100
    similar_texts.append((similarity_percentage, texts[index]))

# Print or use similar_texts as needed

for txt in similar_texts:
  print(txt)
template_job_title = """Instruction:
Extract and provide the job title based on the skills provided below.
Text: {text}
Question: {question}
Output:"""

prompt_job_title = PromptTemplate(template=template_job_title, input_variables=["question", "text"])
chain_job_title = prompt_job_title | llm | StrOutputParser()

similar_texts_str = [txt[1] for txt in similar_texts]  # Extracting only the text from similar_texts

job_titles = chain_job_title.invoke({"question": "Please extract the job title based on the skills provided once without repeating skills.", "text": similar_texts_str},
             config={
    # "callbacks": [ConsoleCallbackHandler()]
})

print(job_titles)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
