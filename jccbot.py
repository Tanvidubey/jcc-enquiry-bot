import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "What's up"],
        "responses": ["Hi there", "Hello", "Hey"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye", "See you later", "Take care"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help"]
    },
    {
        "tag": "help",
        "patterns": ["help", "I need help", "can you help me", "what should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]
    },
    {
        "tag": "course",
        "patterns": ["What courses do you provide", "course", "courses", "what are your courses", "what courses do you offer", "what courses do jcc offer", "what courses do you offer at jcc", "what are your courses at jcc"],
        "responses": ["JCC offers a wide range of courses tailored to meet the demands of today's dynamic job market. From traditional fields like Management and Commerce to emerging sectors like IT and Fashion Technology, JCC provides comprehensive education and training to equip students with the skills and knowledge required to excel in their chosen fields, You can check more information on our website."]
    },
    {
        "tag": "about",
        "patterns": ["about", "about jcc", "tell me about jcc"],
        "responses": ["Jagannath Community College (JCC) is a leading educational institution in Rohini, Delhi. Established with the vision of creating leaders and innovators, JCC offers a diverse range of courses across various disciplines, including Management, Information Technology, Hospitality, Fashion Technology, and more."]
    },
    {
        "tag": "admission",
        "patterns": ["admission", "How can I get admission at jcc", "admission at jcc", "how to get admission", "admission process", "admission process at jcc"],
        "responses": ["The admission process at JCC is rigorous and competitive, ensuring that only the most deserving candidates are selected. It includes a written test, group discussion, and personal interview to assess the candidate's aptitude, skills, and suitability for the chosen course."]
    },
     {
        "tag": "fee",
        "patterns": ["fee structure", "fees", "fee", "what is the fee structure at jcc", "what is the fee structure"],
        "responses": ["JCC offers various courses with different fee structures to accommodate students from diverse backgrounds. The fee structure is designed to be affordable and accessible, ensuring that quality education is within reach for all aspiring students."]
    },
     {
        "tag": "eligibility",
        "patterns": ["eligibility", "eligibility at jcc", "eligibility criteria", "what is the eligibility criteria", "eligibility criteria at jcc"],
        "responses": ["To be eligible for admission at JCC, students must meet certain academic criteria, including securing a minimum of 50% marks in 10+2, including English. Additionally, candidates are evaluated based on their performance in the admission tests and interviews."]
    },
     {
        "tag": "apply",
        "patterns": ["apply", "how to apply", "apply at jcc", "how to apply at jcc"],
        "responses": ["Prospective students can apply for admission to JCC by filling out the online application form available on the college's website or by visiting the campus in person. The application process is straightforward and user-friendly, allowing students to easily navigate through the steps."]
    },
     {
        "tag": "documents",
        "patterns": ["documents", "documents at jcc", "documents required", "documents required at jcc"],
        "responses": ["As part of the admission process, students are required to submit various documents, including their Class XIIth Admit Card, Marksheet & Certificates, Character Certificate, Proof of Residence, Passport size photographs, and Aadhaar Card. These documents help verify the authenticity of the applicant's credentials and facilitate the admission process."]
    },
     {
        "tag": "placement",
        "patterns": ["placement", "placements", "placement at jcc", "placement cell", "placement cell", "placement cell at jcc", "does jcc provide placement"],
        "responses": ["JCC has a dedicated placement cell that assists students in securing internships and job placements."]
    },
     {
        "tag": "contact",
        "patterns": ["contact", "contact at jcc", "contact us", "contact us at jcc"],
        "responses": ["You can contact JCC at admissions@jims.in or call 011-45184100, +91-7617592230."]
    },
     {
        "tag": "online payment",
        "patterns": ["online payment", "online payment at jcc", "online payment options", "online payment options at jcc"],
        "responses": ["One can pay the fee online at https://www.jims.in/online-fee-payment"]
    },
     {
        "tag": "faculty",
        "patterns": ["faculty", "faculty members", "faculty members at jcc", "faculty member", "faculty member at jcc"],
        "responses": ["Our faculty members are highly experienced and qualified professionals with expertise in their respective fields."]
    },
     {
        "tag": "infrastructure",
        "patterns": ["infrastructure", "infrastructure at jcc"],
        "responses": ["JCC has proper infrastructure along with the state-of-the-art classrooms, IT Centre, Communication Lab, Audio-visual equipment and presentation tools, internet and intranet connectivity in classrooms, Digital Electronic Labs and FM Radio Studios, Hospitality Kitchens & Labs, Design Studios and comprehensive library with a hybrid collection of books and journals."]
    },
    {
        "tag": "campus",
        "patterns": ["campus", "jcc campus", "campus at jcc"],
        "responses": ["Our campus is equipped with state-of-the-art facilities, including libraries, laboratories, sports facilities, and more."]
    },
    {
        "tag": "scholarship",
        "patterns": ["scholarship", "does jcc provide any scholarships", "scholarships", "scholarships at jcc"],
        "responses": ["We offer various scholarships and financial aid programs to deserving students. Please visit our website or contact the admission office for more information."]
    },
    {
        "tag": "alumni",
        "patterns": ["alumni", "alumni of jcc", "jcc alumni", "alumnis"],
        "responses": ["Our alumni network is extensive, with graduates excelling in their careers across various industries. Joining our alumni association provides networking opportunities and access to career resources."]
    },
    {
        "tag": "online resources",
        "patterns": ["online resources", "online resources at jcc", "online resources provided", "does jcc provide online resources", "jcc provides online resources or not"],
        "responses": ["We provide access to online resources and learning platforms to enhance the learning experience for our students.",
]
    },
    {
        "tag": "research opportunities",
        "patterns": ["research", "research opportunities", "research opportunities at jcc", "research opportunities provided", "does jcc provide research opportunities", "jcc provides research opportunities or not"],
        "responses": ["Students have access to research opportunities and projects in collaboration with industry partners and research institutions."]
    },
    {
        "tag": "events",
        "patterns": ["events", "events and activities", "events and activities at jcc", "events at jcc", "does jcc organise events and activities", "jcc organises events and activities or not"],
        "responses": ["We organize various events, workshops, and activities to foster a vibrant campus community and enrich the student experience."]
    },
    {
        "tag": "job",
        "patterns": ["job", "jobs", "job opportunities", "job opportunities provided by jcc", "job opportunities provided", "does jcc provide job opportunities", "jcc provides job opportunities or not"],
        "responses": ["We facilitate job placements and internships for our students through our extensive network of industry partners and recruiters."]
    },
    {
        "tag": "fest",
        "patterns": ["fest", "fests", "cultural activities", "cultural activities at jcc", "fests at jcc"],
        "responses": ["We organize various fests and cultural activities to celebrate the heritage of JCC few of them are Verve(annual fest) & sonic(IT fest)."]
    },
    {
        "tag": "location",
        "patterns": ["where is jcc located", "location", "location of jcc", "address"],
        "responses": ["JCC is located in Rohini, Sector-3, Delhi"]
    },
    {
        "tag": "website",
        "patterns": ["website", "website of jcc", "jcc website"],
        "responses": ["https://www.jims.in/"]
    }
] 

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)
    else:
        print("I don't know about that, you can visit our website(https://www.jims.in/) for more information.")

# training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

counter = 0
user_logs = []

def load_user_data(filename="user_data.txt"):
    users = {}
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            for line in file:
                username, password = line.strip().split(":")
                users[username] = password
    return users

def save_user_data(users, filename="user_data.txt"):
    with open(filename, 'w') as file:
        for username, password in users.items():
            file.write(f"{username}:{password}\n")

def main():
    global counter, user_logs

    # Load user data
    users = load_user_data()

    # Session State
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    # Login/Register Form
    if not st.session_state.authenticated:
        st.title("JCC Enquiry Bot")
        login_mode = st.radio("Login/Register:", ("Login", "Register"))
        if login_mode == "Login":
            username = st.text_input("Username:")
            password = st.text_input("Password:", type="password")
            if st.button("Login"):
                if username in users and users[username] == password:
                    st.session_state.authenticated = True
                    st.success("Login Successful!")
                    st.write("Welcome back, " + username + "!")
                else:
                    st.error("Invalid username or password.")
        elif login_mode == "Register":
            new_username = st.text_input("New Username:")
            new_password = st.text_input("New Password:", type="password")
            if st.button("Register"):
                if new_username in users:
                    st.error("Username already exists. Please choose another one.")
                else:
                    users[new_username] = new_password
                    save_user_data(users)
                    st.session_state.authenticated = True
                    st.success("Registration Successful!")
                    st.write("Welcome, " + new_username + "! Please login to continue.")

    # Chat with Bot
    if st.session_state.authenticated:
        st.title("JCC Enquiry Bot")
        st.write("Welcome to the JCC Enquiry Bot. Please type a message and press Enter to start the conversation.")

# Container for chat messages
        chat_container = st.container()

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            with chat_container:
                st.text_area("Chat", value=f"You: {user_input}", height=20, max_chars=None)
                response = chatbot(user_input)
                st.text_area("Chat", value=f"JCC bot: {response}", height=50, max_chars=None)

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

if __name__ == '__main__':
    main()


