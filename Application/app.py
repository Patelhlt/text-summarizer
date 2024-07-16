import nltk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import gradio as gr
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from heapq import nlargest

nltk.download('punkt')
nltk.download('stopwords')

model_path='fine-tuned-abstractive'

# Load the model
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Then load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)


# Define the abstractive summarization function
def abstractive_summary(text,model, tokenizer):
    max_length = 150
    min_length = 30
    num_beams = 4
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=num_beams, early_stopping=True)
    summary =tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Define the extractive summarization function using NLTK
def extractive_summary(text):
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    if len(sentences) < 3:
        return text  # Return original text if it's too short

    # Tokenize text into words
    words = word_tokenize(text.lower())

    # Remove stopwords and non-alphabetic words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalpha() and word not in stop_words]

    # Calculate word frequencies
    freq_dist = FreqDist(words)

    # Score sentences based on word frequencies
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in freq_dist:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = freq_dist[word]
                else:
                    sentence_scores[sentence] += freq_dist[word]

    # Get the top 3 highest scoring sentences
    summary_sentences = nlargest(3, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    return summary

# Combine both summarization methods
def summarize(text, method):
    if method == "Abstractive":
        return abstractive_summary(text, model, tokenizer) # Pass the model and tokenizer to abstractive_summary
    else:
        return extractive_summary(text)

# Define the Gradio interface
iface = gr.Interface(
    fn=summarize,
    inputs=[
        gr.components.Textbox(lines=10, label="Input Text"),  # Use gr.components.Textbox
        gr.components.Radio(["Abstractive", "Extractive"], label="Summarization Method") # Use gr.components.Radio
    ],
    outputs="text",
    title="Text Summarizer",
    description="Enter text and choose summarization method (Abstractive or Extractive)"
)

# Launch the interface
iface.launch(server_name="0.0.0.0", server_port=7860, share=False)
