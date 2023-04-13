import random

import gradio as gr
from src.generate import Generator

generator = Generator()


def main():
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column(scale=1):
                input = gr.Image(type="filepath", label="Image")
                idx = gr.Radio([1, 2, 3, 4, 5, 6, 7, 8], label="Style Index", value=1)
                submit_btn = gr.Button(label="Submit")
            with gr.Column(scale=1):
                output = gr.Image(type="filepath", label="Output")

        input.change(fn=generator.change, inputs=[input], outputs=None)
        submit_btn.click(fn=generator.run, inputs=[idx], outputs=output)

    app.launch(server_name="0.0.0.0", server_port=8070, share=True)


if __name__ == "__main__":
    main()
