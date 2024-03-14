from transformers import AutoTokenizer, pipeline
import torch

# Initialize the model and tokenizer
model = "google/gemma-7b-it"
tokenizer = AutoTokenizer.from_pretrained(model)
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    # model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda"  # Assuming 'cuda:0', adjust if necessary
)

# Initial chat history
chat_history = [
    {"role": "user", "content": "Who are you? Please, answer in pirate-speak."},
]

def interact_with_model(chat_history):
    # Generate the prompt based on the chat history
    prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
    outputs = text_generation_pipeline(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    
    # Extract the model's response from the outputs
    response_text = outputs[0]["generated_text"][len(prompt):].strip()
    
    # Append the model's response to the chat history
    chat_history.append({"role": "model", "content": response_text})
    return response_text

# Interactive loop
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    chat_history.append({"role": "user", "content": user_input})
    model_response = interact_with_model(chat_history)
    print(f"Model: {model_response}")
