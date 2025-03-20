from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
from typing import List
from url_processor import URLProcessor
from qa_model import EnhancedQAModel as QAModel

app = FastAPI()
url_processor = URLProcessor()
qa_model = QAModel()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FastAPI endpoints for backend
@app.post("/process_urls")
async def process_urls(urls: List[str]):
    """Processes URLs and updates model context."""
    try:
        print("\n===== Starting URL Processing =====")
        print(f"Processing {len(urls)} URLs: {urls}")
        result = url_processor.process_urls(urls)
        
        # Extract text content from stored data
        stored_content = url_processor.get_stored_content()
        if not stored_content:
            print("Error: No content was stored from URLs")
            return {"status": "error", "message": "No content was extracted from URLs"}
        
        print(f"\n===== Processing URLs =====")
        print(f"Number of URLs processed: {len(stored_content)}")
        
        # Validate and combine content
        valid_content = []
        total_words = 0
        for url, content in stored_content.items():
            if content:
                clean_content = content.strip()
                words = len(clean_content.split())
                if words >= 10:  # Basic validation
                    total_words += words
                    print(f"\nContent from {url}:")
                    print(f"- Words: {words}")
                    print(f"- Sample: {clean_content[:100]}...")
                    valid_content.append(clean_content)
                else:
                    print(f"Skipping short content from {url} ({words} words)")
            else:
                print(f"Skipping empty content from {url}")
        
        if not valid_content:
            return {"status": "error", "message": "No valid content was extracted from URLs"}
            
        print(f"\nTotal content statistics:")
        print(f"- Valid URLs: {len(valid_content)}")
        print(f"- Total words: {total_words}")
        
        # Join content and validate final result
        # Simple content joining
        content = ". ".join(c.strip().rstrip('.') for c in valid_content) + "."
        if not content or len(content.strip().split()) < 50:
            print("Error: Combined content too short or empty")
            return {"status": "error", "message": "Not enough valid content extracted from URLs"}
            
        print(f"\nFinal content statistics:")
        print(f"- Character length: {len(content)}")
        print(f"- Word count: {len(content.split())}")
        print(f"- Sample: {content[:200]}...")
        
        qa_model.update_context(content)
    except Exception as e:
        print(f"Error processing URLs: {str(e)}")
        return {"status": "error", "message": f"Error processing URLs: {str(e)}"}
    
    return {"status": "success", "processed_urls": len(result)}

@app.post("/ask_question")
async def ask_question(question: str):
    """Returns an answer based on the processed context."""
    try:
        print(f"\n===== Processing Question =====")
        print(f"Question: {question}")
        
        # Validate question
        if not question or len(question.strip()) < 2:
            return {"error": "Please provide a valid question"}
            
        answer = qa_model.get_answer(question)
        print(f"Final answer: {answer}")
        return {"answer": answer}
    except Exception as e:
        print(f"Error processing question: {str(e)}")
        return {"error": f"Error processing question: {str(e)}"}

# Gradio interface
def process_urls_gradio(urls_text: str):
    urls = [url.strip() for url in urls_text.split("\n") if url.strip()]
    result = url_processor.process_urls(urls)

    # Extract content correctly
    content = "\n".join(url_processor.get_stored_content().values())  # 🚀 FIXED
    qa_model.update_context(content)
    
    return "URLs processed successfully!"

def ask_question_gradio(question: str):
    """Gradio interface for asking questions."""
    try:
        print(f"\n===== Processing Gradio Question =====")
        print(f"Question: {question}")
        
        # Validate question
        if not question or len(question.strip()) < 2:
            return "Please provide a valid question"
            
        answer = qa_model.get_answer(question)
        print(f"Final answer: {answer}")
        return answer
    except Exception as e:
        print(f"Error processing question: {str(e)}")
        return f"Error processing question: {str(e)}"

# Create Gradio interface
with gr.Blocks() as interface:
    gr.Markdown("# Web Content Q&A Tool")

    with gr.Tab("Process URLs"):
        urls_input = gr.Textbox(label="Enter URLs (one per line)", lines=5)
        process_button = gr.Button("Process URLs")
        url_output = gr.Textbox(label="Status")
        process_button.click(
            fn=process_urls_gradio,
            inputs=urls_input,
            outputs=url_output
        )

    with gr.Tab("Ask Questions"):
        question_input = gr.Textbox(label="Enter your question")
        ask_button = gr.Button("Ask Question")
        answer_output = gr.Textbox(label="Answer")
        ask_button.click(
            fn=ask_question_gradio,
            inputs=question_input,
            outputs=answer_output
        )

# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, interface, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
