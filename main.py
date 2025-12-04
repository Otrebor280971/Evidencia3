import gradio as gr

def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

demo = gr.Interface(
    fn=greet,
    inputs=["text", "button"],
    outputs=["text"],
    api_name="predict"
)

demo.launch()