import time
import uuid
import html
import threading
import numpy as np
import gradio as gr
from queue import Queue
from tokenizer import tokenizer,vocab_size,token2str

import torch
import torch.nn as nn
from make_model import make_model
from train_and_use import El_text_continue_stream
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

model = make_model(
    #token是从1开始的，0填充，剩下的用来覆盖全部字节
    vocab_size = vocab_size+1+255,
    embedding_dim = 768,
    key_dim = 128,
    head_number = 12,
    position_information_type = "mask",
    enable_affine = True,
    enable_talking_head = True,
    use_diff = False,
    self_attention_block_size = 0,
    feed_forward_dim = 1536,
    enable_layer_norm = True,
    deep = 12,
    dropout_rate = 0.1,
    enable_el_cache = True
).to(device)
model.load_state_dict(torch.load('large_model_13033051.weight',weights_only=True))
model = model.eval()

# 全局字典，存 per-session 的不可 deepcopy 对象 / 状态
user_queues = {}             # session_id -> Queue()
user_stop_flags = {}         # session_id -> bool (True 表示停止)
user_current_sessions = {}   # session_id -> 最后一个 session (list), 可 deepcopy

# token包装器
def token_wapper(token):
    return f'<span style="background-color: #FFD580; padding: 2px 4px; border-radius: 6px; margin: 1px; display: inline-block;">{html.escape(token)}</span>'

def token_split_wapper(token):
    return f'<span style="background-color: #FF0000; padding: 2px 4px; border-radius: 6px; margin: 1px; display: inline-block;">{html.escape("["+token+"](单字多token)")}</span>'

# 后台生成函数（只访问全局字典，通过 session_id 定位）
def generate_text(user_message, session_id, temperature, repeat_penalty, max_length, decay):
    out = ""
    q = user_queues.get(session_id)
    if q is None:
        return
    # 通过分词器转化为token
    user_tokens = tokenizer(user_message,5.0)
    # 将token还原并进行包装
    words = []
    temp = []
    for token in user_tokens:
        if token > 0:
            if len(temp):
                words += [token_split_wapper(token2str(temp))]
                temp = []
            words += [token_wapper(token2str([token]))]
        else:
            temp += [token]
    if len(temp):
        words += [token_split_wapper(token2str(temp))]
    user_tokens = ''.join(words)
    # 准备模型输入
    if len(tokenizer(user_message,5.0)) < 2:
        user_message = f' {user_message}'
    tokens_batch = [tokenizer(user_message,5.0)]
    tokens_batch = np.array(tokens_batch,dtype=np.int64)+255
    inputs = torch.from_numpy(tokens_batch).to(device).data
    last_len = -1
    # 模型输出
    with torch.no_grad():
        for o in El_text_continue_stream(
            model,inputs,out_length=max_length,
            repeat_penalty_value=repeat_penalty,
            temperature=temperature,
            decay=decay
        ):
            split = ''
            if o[0,-1] > 255: #确保是完整的字符才可以输出
                out += token_wapper(token2str(o[0][last_len:].cpu().numpy()-255,split=split))
                last_len = -1
                sess = [
                    {"role": "user", "content": user_tokens},
                    {"role": "assistant", "content": out},
                ]
                user_current_sessions[session_id] = sess
                try:
                    q.put(sess, block=False)
                except:
                    # 极少情况：队列放入失败（一般不会发生），忽略
                    pass
            else:
                last_len -= 1
            if user_stop_flags.get(session_id, True):
                break

# 点击按钮的处理逻辑：start / stop / clear
def click_process(sess, label, user_message, state, stop_flag_state, session_id, temperature, repeat_penalty, max_length, decay):
    # 安全检查
    if session_id is None or session_id not in user_queues:
        # session 还没初始化好，直接返回不改变 UI
        return "", "发送消息", state or {"current_session": []}, stop_flag_state or {"stop": True}, session_id

    # 如果现在处于"停止"状态并且有用户输入 -> 启动生成线程
    if stop_flag_state.get("stop", True) and user_message and sess == []:
        user_stop_flags[session_id] = False
        thread = threading.Thread(target=generate_text, args=(user_message, session_id, temperature, repeat_penalty, max_length, decay))
        thread.daemon = True
        thread.start()
        # 更新返回给前端的 state/stop_flag（gradio 会把这些值保存到 session state）
        return "",  "终止输出", {"current_session": user_current_sessions.get(session_id, [])}, {"stop": False}, session_id

    # 如果正在输出 -> 终止
    elif not stop_flag_state.get("stop", True) and label != "清空会话":
        user_stop_flags[session_id] = True
        return user_message, "清空会话", {"current_session": user_current_sessions.get(session_id, [])}, {"stop": True}, session_id

    # 否则清空会话
    else:
        user_stop_flags[session_id] = True
        user_current_sessions[session_id] = []
        q = user_queues.get(session_id)
        if q:
            while not q.empty():
                try:
                    q.get_nowait()
                except:
                    break
        return user_message, "发送消息", {"current_session": []}, {"stop": True}, session_id

# 流式输出 generator（只需触发一次即可一直运行）
def stream_output(state, stop_flag_state):
    global user_queues,user_stop_flags,user_current_sessions
    # 页面加载时初始化 session（返回可 deepcopy 的 state 值和 session_id）
    session_id = str(uuid.uuid4())
    user_queues[session_id] = Queue()
    user_stop_flags[session_id] = True
    user_current_sessions[session_id] = []  # 初始为空会话
    # 返回给 gradio 的 state 值（这些都是 deepcopy-friendly）
    yield gr.update(), gr.update(), {"current_session": []}, {"stop": True}, session_id
    t0 = time.time()
    while True:
        q = user_queues[session_id]
        stopped = user_stop_flags.get(session_id, True)
        # 优先处理队列中的消息（FIFO）
        if (not stopped) and (not q.empty()):
            t0 = time.time()
            while q.qsize() > 5:
                sess = q.get()
            sess = q.get()
            # 更新 chatbot（返回的 sess 是 [{"role":...}, ...]）
            # 同时把 state 返回为 deepcopy-friendly 字典（gr.State 需要可 deepcopied）
            yield sess, "终止输出", {"current_session": sess}, gr.update(), session_id
        else:
            last = user_current_sessions.get(session_id, [])
            if last == []:
                yield last, gr.update(), {"current_session": last},gr.update(), session_id
            else:
                if time.time() - t0 > 1:
                    yield last, "清空会话", {"current_session": last},gr.update(), session_id
        time.sleep(0.01)  # 防止 busy-wait 占满 CPU

# ========== Gradio UI ==========
with gr.Blocks() as demo:
    gr.Markdown("# LLM 在线体验")
    
    chatbot = gr.Chatbot(type="messages", label="输入/输出", autoscroll=False, show_copy_button=False)
    msg = gr.Textbox(placeholder="请输入需要续写的句子，比如：今天天气真好", label="输入需要补全的句子", lines=4)
    
    with gr.Row():
        temperature = gr.Slider(0.0001, 3.0001, value=0.0001, step=0.1, label="Temperature")
        repeat_penalty = gr.Slider(0.0, 5.0, value=2.5, step=0.1, label="Repeat Penalty")
        max_length = gr.Slider(64, 8192, value=128, step=64, label="Max Length")
        decay = gr.Slider(0.90, 1.0, value=0.98, step=0.01, label="Repeat Penalty Decay Rate")
    
    btn = gr.Button("发送消息")

    # gr.State 用来在前端保存可 deepcopied 的 session 值
    state = gr.State()
    stop_flag_state = gr.State()
    session_id = gr.State()

    # 点击按钮处理 - 使用 session_id 定位用户资源
    btn.click(
        click_process,
        inputs=[chatbot, btn, msg, state, stop_flag_state, session_id, temperature, repeat_penalty, max_length, decay],
        outputs=[msg, btn, state, stop_flag_state, session_id],
    )

    # 页面加载后再触发 stream_output（只要触发一次，generator 会一直运行）
    demo.load(
        stream_output,
        inputs=[state, stop_flag_state],
        outputs=[chatbot, btn, state, stop_flag_state, session_id],
    )

if __name__ == "__main__":
    demo.queue(max_size=128, default_concurrency_limit=128)
    demo.launch(share=False)