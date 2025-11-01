from agent import SmartTravelAgent, WeatherService
import gradio as gr

# --- Initialize agent ---
agent = SmartTravelAgent("travel_data.json")
agent.weather = WeatherService(api_key="e3e562385469e9591bb7cf6e1ab9cb92")

# --- Chat logic ---
def respond(msg, hist):
    if not msg:
        return hist
    reply = agent.chat(msg)
    hist.append({"role": "user", "content": msg})
    hist.append({"role": "assistant", "content": reply})
    return hist

def clear():
    agent.memory.clear()
    return []

# --- UI ---
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Vietnam Travel Agent",
    css="""
    @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap');

    * {
        font-family: 'Be Vietnam Pro', sans-serif !important;
    }
    #chatbot {
        font-size: 16px;
        line-height: 1.6;
    }
    .gr-button {
        font-weight: 600 !important;
        border-radius: 10px !important;
        padding: 8px 16px !important;
    }
    h1, h2, h3 {
        font-weight: 700;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        opacity: 0.7;
        margin-top: 12px;
    }
    """
) as demo:

    gr.Markdown(
        """
        # Vietnam Travel Agent by Phương Anh
        *Ask me about travel, food, weather, or costs — in English or Vietnamese*
        *Hỏi tôi về du lịch, món ăn, thời tiết hoặc chi phí nhaaa!*
        """
    )

    # Suggested Questions
    with gr.Row():
        b1 = gr.Button("Tell me about Hanoi")
        b2 = gr.Button("What to eat in Da Nang?")
        b3 = gr.Button("Best time to visit Phu Quoc?")
        b4 = gr.Button("Thời tiết Hà Nội lúc này?")
        b5 = gr.Button("Chi phí đi Hồ Chí Minh?")

    chatbot = gr.Chatbot(type="messages", height=500, elem_id="chatbot")

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Type your question... / Nhập câu hỏi...",
            show_label=False,
            scale=4
        )
        send = gr.Button("Send / Gửi", variant="primary")
        clear_btn = gr.Button("Clear Chat")

    # Bind logic
    send.click(respond, [msg, chatbot], chatbot).then(lambda: "", None, msg)
    msg.submit(respond, [msg, chatbot], chatbot).then(lambda: "", None, msg)
    clear_btn.click(clear, outputs=chatbot)

    # Suggested question bindings
    b1.click(respond, [b1, chatbot], chatbot)
    b2.click(respond, [b2, chatbot], chatbot)
    b3.click(respond, [b3, chatbot], chatbot)
    b4.click(respond, [b4, chatbot], chatbot)
    b5.click(respond, [b5, chatbot], chatbot)

    gr.Markdown("<div class='footer'>by Phương Anh</div>")

# --- Launch ---
if __name__ == "__main__":
    demo.launch(debug=True)
