import gradio as gr
import requests

def chat_with_model(user_input, chat_history):
    headers = {
        "Content-Type": "application/json",
    }
    data = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input}
        ]
    }
    response = requests.post("http://localhost:8000/v1/chat/completions", headers=headers, json=data)
    response_json = response.json()
    assistant_message = response_json['choices'][0]['message']['content']
    chat_history.append((user_input, assistant_message))
    return chat_history, chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    with gr.Row():
        txt = gr.Textbox(show_label=False, container=False, placeholder="Type your prompt here...")
        txt.submit(chat_with_model, [txt, chatbot], [chatbot, chatbot])

demo.launch()

