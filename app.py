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
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from eli5.lime import TextExplainer
    from eli5.lime.samplers import MaskingTextSampler
    import eli5

    # Initialize Spacy
    nlp = spacy.load("en_core_web_sm")
    
    # device to run DL model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    def classify_text(model, text):
        # Clean and format the input text
        text = format_text(text)
        
        # Predict the class of the input text
        if type(model) == "sklearn.pipeline.Pipeline":
            # Predict using either LR or NB
            prediction = model.predict([text]).item()
            st.session_state["sklearn"] = True
        else:
            # DL models (BERT/DistilBERT based models)
            cleaned_text_tokens = tokenizer([text], padding='max_length', max_length=512, truncation=True)
            with torch.inference_mode():
                input_ids, att = cleaned_text_tokens["input_ids"], cleaned_text_tokens["attention_mask"]
                input_ids = torch.tensor(input_ids).to(device)
                attention_mask = torch.tensor(att).to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)[0]
                _, prediction = torch.max(logits, 1)
                prediction = prediction.item()
        
        # Map the predicted class to string output
        if prediction == 0:
            return "Human-made 🤷‍♂️🤷‍♀️"
        else:
            return "Generated with AI 🦾🤖"

    @st.cache(allow_output_mutation=True, suppress_st_warning=True)
    def load_tokenizer(option):
        if option == "BERT-based model":
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        else:
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        return tokenizer

    @st.cache(allow_output_mutation=True, suppress_st_warning=True)
    def load_model(option):
        model = AutoModelForSequenceClassification.from_pretrained(models_available[option])
        return model
    

    # Streamlit app:

    models_available = {"Logistic Regression":"models/baseline_model_lr2.joblib", 
                        "Naive Bayes": "models/baseline_model_nb2.joblib",
                        "DistilBERT-based model (BERT light)": "ferdmartin/HF_DistilBertBasedModelAppDocs",
                        "BERT-based model": "ferdmartin/HF_BertBasedModelAppDocs"}

    st.set_page_config(layout="wide")
    st.title("Academic Application Document Classifier")
    st.header("Is it human-made 📝 or Generated with AI 🤖 ?  ")
    
    # Check the model to use
    option = st.selectbox("Select a model to use:", models_available)

    # Load the selected trained model
    if option in ("BERT-based model", "DistilBERT-based model (BERT light)"):
        tokenizer = load_tokenizer(option)
        model = load_model(option)
    else:
        model = joblib.load(models_available[option])


    text = st.text_area("Enter either a statement of intent or a letter of recommendation:")

    #Hide footer "made with streamlit"
    hide_st_style = """
            <style>
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    # Use model
    if st.button("Let's check this text!"):
        if text.strip() == "":
            st.error("Please enter some text")
        else:
            # Add a progress bar
            progress_bar = st.progress(0)

            # Add a placeholder for the progress message
            status_text = st.empty()

            # Simulate a long-running process
            for i in range(100):
                # Update the progress bar every 0.02 seconds
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
            prediction = classify_text(model, text)
            # Store the result in session state
            st.session_state["prediction"] = prediction
            st.session_state["text"] = text
            
            
            # Print result
            st.write(f"<span style='font-size: 24px;'>I think this text is: {prediction}</span>", 
                    unsafe_allow_html=True)

    elif "prediction" in st.session_state:
        # Display the stored result if available
        st.write(f"<span style='font-size: 24px;'>I think this text is: {st.session_state['prediction']}</span>", 
                unsafe_allow_html=True)

    if st.button("Model Explanation"):
        # Check if there's text in the session state
        if "text" and "sklearn" in st.session_state:
            with st.spinner('Wait for it...'):
                explainer = TextExplainer(sampler=MaskingTextSampler())
                explainer.fit(st.session_state["text"], model.predict_proba)
                html = eli5.format_as_html(explainer.explain_prediction(target_names=["Human", "AI"]))
                # Render HTML
                st.components.v1.html(html, height=500,scrolling = True)
        else:
            st.error("Please enter some text and click 'Let's check!' before requesting an explanation.")
            
if __name__ == "__main__":
    main()
