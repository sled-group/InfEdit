import gradio as gr

canvas_html = """</script><rich-text-editor id="rich-text-root"></rich-text-editor>"""
load_js = """
async () => {
  // add scripts on Gradio load
  const scripts = ["https://cdn.quilljs.com/1.3.6/quill.min.js","file=rich-text-to-json.js"]
  scripts.forEach(src => {
    const script = document.createElement('script');
    script.src = src;
    document.head.appendChild(script);
  })
}
"""
get_js_data = """
async (richTextData) => {
  const richEl= document.getElementById("rich-text-root");
  const data = richEl? richEl._data : null;
  return data
}
"""


def predict(rich_text_data):
    print(rich_text_data)

    return rich_text_data


blocks = gr.Blocks()
with blocks:
    rich_text_data = gr.JSON(value={}, visible=False)
    with gr.Row():
        with gr.Column(visible=True) as box_el:
            rich_text_el = gr.HTML(canvas_html, elem_id="canvas_html")
        with gr.Column(visible=True) as box_el:
            data_out = gr.JSON()

    btn = gr.Button("Run")
    btn.click(fn=predict, inputs=[rich_text_data],
              outputs=[data_out], _js=get_js_data)
    blocks.load(None, None, None, _js=load_js)

blocks.launch(debug=True, inline=True)
