import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

class EnhancedQAModel:
    def __init__(self):
        model_name = "deepset/roberta-base-squad2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.context = ""
        
    def update_context(self, new_context: str):
        """Update the context used for answering questions."""
        self.context = new_context.strip()

    def get_answer(self, question: str) -> str:
        """Get an answer for the given question using the stored context."""
        if not self.context.strip():
            return "Please provide context before asking questions."

        question = question.strip()
        if not question.endswith('?'):
            question += '?'

        # Tokenize full context
        inputs = self.tokenizer(self.context, add_special_tokens=True, return_tensors="pt", truncation=False)
        input_ids = inputs["input_ids"].squeeze(0)

        # Handle long contexts with larger chunks and more overlap
        chunk_size = 512  # Increased from 384
        stride = 256  # Increased from 128 for better context retention
        chunks = []
        
        # Create overlapping chunks
        for i in range(0, len(input_ids), chunk_size - stride):
            chunk = input_ids[i:i + chunk_size]
            if len(chunk) < 2:
                break
            chunks.append(chunk)

        max_score = -float('inf')
        best_answer = ""
        no_answer_score = -float('inf')

        try:
            # Process all chunks for better coverage
            for chunk in chunks:
                if len(chunk) < 10:
                    continue
                    
                # Prepare input for model
                question_inputs = self.tokenizer(
                    question,
                    self.tokenizer.decode(chunk),
                    add_special_tokens=True,
                    return_tensors="pt",
                    max_length=512,
                    truncation='longest_first',
                    padding=True,
                    return_overflowing_tokens=False
                )
                
                # Get model outputs
                with torch.no_grad():
                    outputs = self.model(**question_inputs)
                    
                start_scores = outputs.start_logits[0]
                end_scores = outputs.end_logits[0]
                
                # Find best answer in this chunk
                for start_idx in range(len(start_scores)):
                    for end_idx in range(start_idx, min(start_idx + 50, len(end_scores))):
                        score = start_scores[start_idx] + end_scores[end_idx]
                        
                        if score > max_score:
                            max_score = score
                            answer_span = question_inputs["input_ids"][0][start_idx:end_idx + 1]
                            candidate_answer = self.tokenizer.decode(answer_span)
                            
                            # Basic answer validation
                            if len(candidate_answer.split()) >= 2 and len(candidate_answer) >= 10:
                                best_answer = candidate_answer

            # Clean and validate the answer
            if best_answer and max_score > -1:
                # Remove special tokens and clean up
                best_answer = best_answer.replace("[CLS]", "").replace("[SEP]", "").strip()
                best_answer = " ".join(best_answer.split())
                
                # Basic answer validation
                if len(best_answer) < 5 or len(best_answer.split()) < 2:
                    return "I could not find a specific answer to your question in the provided context."
                    
                return best_answer
            else:
                return "I could not find a specific answer to your question in the provided context. Please try rephrasing your question or providing more relevant context."

        except Exception as e:
            return f"An error occurred while processing your question: {str(e)}"