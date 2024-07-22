from .embedder import VectorDB, get_embeddings
from .utils import read_problems, problems_filenames
from tqdm.auto import tqdm
import gradio as gr
import json
from openai import OpenAI

db = VectorDB().load()
emb_keys = set([x[0] for x in db.metadata])
print("read", len(emb_keys), "embeddings from db")
problems = {}
for f in problems_filenames():
    for p in read_problems("problems/" + f):
        problems[p["uid"]] = p
print("read", len(problems), "problems from db")

with open("settings.json", encoding='utf-8') as f:
    settings = json.load(f)

client = OpenAI(
    api_key=settings["OPENAI_API_KEY"],
)


def querier(statement, template_choice, topk):
    # print(statement, template_choice)
    paraphrased = statement
    if "None" not in template_choice:
        template_id = int(template_choice.split(" ")[1]) - 1
        template = settings["TEMPLATES"][template_id]
        ORIGINAL = "\n" + statement + "\n"
        prompt = template.replace("[[ORIGINAL]]", ORIGINAL).strip()
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            timeout=120,
            max_tokens=4000,
            model="gpt-4o",
        )
        assert chat_completion.choices[0].finish_reason == "stop"
        paraphrased = chat_completion.choices[0].message.content
    # emb = get_embeddings([paraphrased])[0] # 临时删除test
    # query nearest
    nearest = db.query_nearest((get_embeddings([paraphrased])[0]), k=topk)
    # print(nearest)
    return paraphrased, {b[1]: a for a, b in nearest}


def show_problem(evt: gr.SelectData):  # SelectData is a subclass of EventData
    uid = evt.value
    statement = problems[uid]["statement"].replace("\n", "\n\n")
    summary = sorted(problems[uid]["processed"], key=lambda t: t["template_md5"])
    if len(summary):
        summary = summary[0]["result"]
    else:
        summary = None
    title = uid  # problems[uid]['title']
    url = problems[uid]["url"]
    markdown = f"# [{title}]({url})\n\n"
    if summary is not None:
        markdown += f"### Summary (auto-generated)\n\n{summary}\n\n"
    markdown += f"### Statement\n\n{statement}"
    return markdown


with gr.Blocks(
    title="查重网-信息学题目重复检测器", css=".mymarkdown {font-size: 15px !important}"
) as demo:
    gr.Markdown(
        """
    # 查重网-信息学题目重复检测器
    
    ### 用于竞争性编程问题的语义搜索引擎，用于查询信竟中出的题目是否重复（永久免费）。
    ------
    感谢所有我能找到的校内OJ、公开OJ，没有出题人就不会有这个查重系统——对了，该系统由 [QNXS Dog](https://www.qnxsdog.com) 开发~
    
    ### 使用说明
    * 将你的题目放到框里，然后选择一种提示词，点击提交即可
    * 一般会在两分钟之内返回结果【当前队列人数：> 10】，返回列表的 % 代表相似度，一般非原题不超过 30%
    * 如果超过 30% 建议自行检查是否原题，不要依赖本工具，有可能有误报情况~
    """
    )
    with gr.Row():
        # column for inputs
        with gr.Column():
            input_text = gr.Textbox(
                label="你的题目",
                info="把你出的题目复制在这个框中，不超过4K长度！",
                value="计算输入序列的最长递增子序列。",
            )
            template_type = gr.Radio(
                # ["新版（翻译成中文，推荐）"] + ["新版（翻译成英文）"] + ["旧版（翻译成英文，不推荐）"] + ["旧版（翻译成中文，不推荐）"] + ["空提示词（快速模式，效果差）"],
                ["Template " + str(x + 1) for x in range(len(settings["TEMPLATES"]))]
                + ["None (faster)"],
                label="使用哪种模式？",
                value="Template 1",
            )
            topk_slider = gr.Slider(
                minimum=1,
                maximum=100,
                step=1,
                value=10,
                label="显示相似问题的数量",
            )
            submit_button = gr.Button("提交【当前队列人数：> 10】")
            my_markdown = gr.Markdown(
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False},
                    {"left": "\\(", "right": "\\)", "display": False},
                    {"left": "\\[", "right": "\\]", "display": True},
                ],
                elem_classes="mymarkdown",
            )
        # column for outputs
        with gr.Column():
            output_text = gr.Textbox(
                label="简化题意",
                value="",
            )
            output_labels = gr.Label(
                label="疑似重复的题目",
            )
    gr.Markdown(
        """
    ### OJ列表：
    * [洛谷-计算机科学教育新生态（包含主题库、B题库）](https://www.luogu.com.cn)
    * [Codeforces](https://www.codeforces.com)
    * [Atcoder](https://atcoder.jp)
    * [Hydro - BZOJ](https://hydro.ac/d/bzoj/)
    * [NKOJ（校内OJ不便给出链接）]()
    * [YCOJ（校内OJ不便给出链接）]()
    * [LLONG OJ](https://oj.llong.tech)
    * [Hydro OJ](https://hydro.ac/d/)

    ### 封禁列表
    * e.d. IP：，封禁原因：，现已查明姓名：、洛谷用户名：
    """
    )
    submit_button.click(
        fn=querier,
        inputs=[input_text, template_type, topk_slider],
        outputs=[output_text, output_labels],
    )
    output_labels.select(fn=show_problem, inputs=None, outputs=[my_markdown])

demo.launch()
