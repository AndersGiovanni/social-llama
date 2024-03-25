from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize the model and tokenizer
model_id = "google/gemma-7b-it"
dtype = torch.bfloat16
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    # torch_dtype=dtype,
)

# Initial chat history
chat = [
    {"role": "user", "content": "Write a hello world program"},
]

def interact_with_model(chat):
    # Generate the prompt based on the chat history
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generate the response from the model
    output = model.generate(**inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Append the model's response to the chat history
    chat.append({"role": "model", "content": response})
    return response

# Interactive loop
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    chat.append({"role": "user", "content": user_input})
    model_response = interact_with_model(chat)
    print(f"Model: {model_response}")
