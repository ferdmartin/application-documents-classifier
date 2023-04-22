def main():
    """
    Creates a Streamlit web app that classifies a given body of text as either human-made or AI-generated,
    using a pre-trained model. 
    """
    import streamlit as st
    import joblib
    import string
    import time
    import spacy
    import re
    from eli5.lime import TextExplainer
    from eli5.lime.samplers import MaskingTextSampler

    # Initialize Spacy
    nlp = spacy.load("en_core_web_sm")


    def format_text(text: str) -> str:
        """
        This function takes a string as input and returns a formatted version of the string. 
        The function replaces specific substrings in the input string with empty strings, 
        converts the string to lowercase, removes any leading or trailing whitespace, 
        and removes any punctuation from the string. 
        """

        text = nlp(text)
        text = " ".join([token.text for token in text if token.ent_type_ not in ["PERSON", "DATE"]])

        pattern = r"\b[A-Za-z]+\d+\b"
        text = re.sub(pattern, "", text)
        
        return text.replace("REDACTED", "").lower().replace("[Name]", "").replace("[your name]", "").\
                                replace("dear admissions committee,", "").replace("sincerely,","").\
                                replace("[university's name]","fordham").replace("dear sir/madam,","").\
                                replace("– statement of intent  ","").\
                                replace('program: master of science in data analytics  name of applicant:    ',"").\
                                replace("data analytics", "data science").replace("| \u200b","").\
                                replace("m.s. in data science at lincoln center  ","").\
                                translate(str.maketrans('', '', string.punctuation)).strip().lstrip()

    # Define the function to classify text
    def classify_text(text):
        # Clean and format the input text
        text = format_text(text)
        
        # Predict the class of the input text
        prediction = model.predict([text])
        
        # Map the predicted class to a string
        if prediction[0] == 0:
            return "Human-made 🤷‍♂️🤷‍♀️"
        else:
            return "Generated with AI 🦾🤖"

    # Streamlit app
    models_available = {"Logistic Regression":"models/baseline_model_lr2.joblib", 
                        "Naive Bayes": "models/baseline_model_nb2.joblib"}

    st.set_page_config(layout="wide")
    st.title("Academic Application Document Classifier")
    st.header("Is it human-made 📝 or Generated with AI 🤖 ?  ")
    
    # Check the model to use
    option = st.selectbox("Select a model to use:", models_available)
    # Load the selected trained model
    model = joblib.load(models_available[option])
    
    text = st.text_area("Enter a statement of intent:")

    # Hide footer "made with streamlit"
    hide_st_style = """
            <style>
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    # Use model
    if st.button("Let's check!"):
        if text.strip() == "":
            st.error("Please enter some text")
        else:
            # Add a progress bar
            progress_bar = st.progress(0)

            # Add a placeholder for the progress message
            status_text = st.empty()

            # Simulate a long-running process
            for i in range(100):
                # Update the progress bar every 0.1 seconds
                time.sleep(0.05)
                progress_bar.progress(i + 1)
                
                if i % 2 == 0:
                    magic = "✨"
                else:
                    magic = ""
                # Update the progress message
                status_text.write(f"Work in progress {i + 1}%... Wait for the magic 🪄🔮{magic}")
            # Clear the progress bar and status message
            progress_bar.empty()
            status_text.empty()

            # Use model
            prediction = classify_text(text)
            # Print result
            st.write(f"<span style='font-size: 24px;'>I think this text is: {prediction}</span>", 
                    unsafe_allow_html=True)
            
            if st.button("Output Explanation"):
                explainer = TextExplainer(sampler=MaskingTextSampler())
                explainer.fit(text, model.predict_proba)
                st.write(explainer.show_prediction(target_names=["Human", "AI"]).data, unsafe_allow_html=True)
            
if __name__ == "__main__":
    main()
