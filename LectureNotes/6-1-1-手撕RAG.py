import PyPDF2
from rank_bm25 import BM25Okapi
import jieba

from sentence_transformers import SentenceTransformer
xiaobu_embed_model = SentenceTransformer('/xxx/maple77/xiaobu-embedding-v2')

import json
import requests
def pdf_to_text(pdf_path):
    pdf2word_page = []
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            pdf2word_page.append({
        'page': 'page_' + str(page_num + 1),
        'content': page.extract_text()
    })
    return pdf2word_page

def bm25_search(question_text, bm25_model):
    doc_scores = bm25_model.get_scores(jieba.lcut(question_text))
    scores_dict = {pos: soc for pos, soc in enumerate(doc_scores)}
    
    # 过滤掉是0分的,然后排序
    scores_dict_filter0 = dict(filter(lambda x: x[1] != 0, scores_dict.items()))

    if scores_dict_filter0:
        sorted_scores = sorted(scores_dict_filter0.items(), key=lambda item: item[1], reverse=True)
        nm25_max_score_page_idx_10 = [idx for idx, score in sorted_scores[:10]]
    else:
        nm25_max_score_page_idx_10 = []
    return nm25_max_score_page_idx_10

def xiaobu_model(pdf2word_page):
    pdf_content_sentences = [x['content'] for x in pdf2word_page]
    pdf_embeddings = xiaobu_embed_model.encode(pdf_content_sentences, normalize_embeddings=True)
    return pdf_embeddings

def xiaobu_model_search(question_text, pdf2word_page):
    pdf_embeddings = xiaobu_model(pdf2word_page)
    question_embeddings = xiaobu_embed_model.encode(question_text, normalize_embeddings=True)
    score = question_embeddings @ pdf_embeddings.T
    m3e_max_score_page_idx_10 = score.argsort()[::-1][:10]
    return m3e_max_score_page_idx_10

def fusion_sort(m3e_top, bm_top):
    fusion_score = {}
    k = 60
    for idx, q in enumerate(bm_top):
        if q not in fusion_score:
            fusion_score[q] = 1 / (idx + k)
        else:
            fusion_score[q] += 1 / (idx + k)

    for idx, q in enumerate(m3e_top):
        if q not in fusion_score:
            fusion_score[q] = 1 / (idx + k)
        else:
            fusion_score[q] += 1 / (idx + k)

    sorted_dict = sorted(fusion_score.items(), key=lambda item: item[1], reverse=True)
    return sorted_dict

def LLM_chat(query, studentbook_content = []):
    if not studentbook_content:
        return '未提供相关资料'

    content = {
  "model": "qwen2.5:14b",
    "stream": False,
  "messages": [
    {
      "role": "system",
      "content": '''
你是xxx，负责从提供的xxx中回答xxx提出的问题。请遵循以下原则：
1.xxx
2.xxx
3.xxx
请务必遵循以上原则进行回答。
'''},
    {"role": "user",
      "content": f"仔细阅读以下xxx中的内容：{studentbook_content}\,请回答xxx的问题，问题是:{query}。"
        }]}
    
    
    json_data = json.dumps(content)
    
    url = 'http://xxxxxxxx'
    result = requests.post(url
                     , data = json_data)

    data = result.content.decode('utf-8')
    response = json.loads(data)['message']['content']

    return response

def rag_q2a(query):
    bm_top = bm25_search(query, bm25_model)
    m3e_top = xiaobu_model_search(query, pdf2word_page)
    sorted_dict = fusion_sort(m3e_top, bm_top)
    sorted_dict_idx = [idx for idx, score in sorted_dict]

    prompt_dict_idx = sorted_dict_idx[:8]
    prompt_dict_idx.sort()

    print(prompt_dict_idx)

    prompt_text = ""
    for i in range(len(prompt_dict_idx)):
        current_idx = prompt_dict_idx[i]
        prev_idx = prompt_dict_idx[i - 1] if i > 0 else None
        next_idx = prompt_dict_idx[i + 1] if i < len(prompt_dict_idx) - 1 else None

        if prev_idx is not None and current_idx == prev_idx + 1:
            # 如果当前段落和前一个段落相邻，只拼接当前段落的剩余部分
            prev_content = ""
            current_content = pdf2word_page[current_idx]['content'][50:]
        else:
            prev_content = pdf2word_page[prev_idx]['content'][-50:] if prev_idx is not None else ""
            current_content = pdf2word_page[current_idx]['content']

        next_content = pdf2word_page[next_idx]['content'][:50] if next_idx is not None else ""

        prompt_text += prev_content + current_content + next_content + '\n\n----分割页-----\n\n'

    rag_res = LLM_chat(query, prompt_text)
    return rag_res, prompt_text