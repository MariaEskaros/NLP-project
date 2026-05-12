def build_context(retrieved_chunks):
    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            f"[Source: {chunk['episode']} | Rank: {chunk['rank']}]\n"
            f"{chunk['text']}"
        )

    return "\n\n---\n\n".join(context_parts)


def build_minimal_prompt(question, retrieved_chunks):
    context = build_context(retrieved_chunks)

    return f"""
Context:
{context}

Question:
{question}

Answer:
""".strip()


def build_strict_grounded_prompt(question, retrieved_chunks):
    context = build_context(retrieved_chunks)

    return f"""
You are an Arabic question-answering assistant.

You must answer using ONLY the provided context.

Rules:
1. Do not use outside knowledge.
2. If the answer is not clearly found in the context, say:
   "لا توجد معلومات كافية في النص للإجابة عن هذا السؤال."
3. Answer in the same language as the user's question.
4. If the question mixes Arabic and English, you may answer in Arabic with necessary English terms.
5. Keep the answer clear and concise.
6. Mention the source episode name at the end.

Retrieved context:
{context}

User question:
{question}

Grounded answer:
""".strip()


def build_arabic_strict_prompt(question, retrieved_chunks):
    context = build_context(retrieved_chunks)

    return f"""
انت مساعد ذكي للاجابة على الاسئلة بناء على نصوص عربية مسترجعة.

يجب ان تلتزم بالقواعد التالية:
1. اجب فقط من السياق المسترجع.
2. لا تضف معلومات من خارج النص.
3. اذا لم تكن الاجابة موجودة بوضوح في السياق، قل:
   "لا توجد معلومات كافية في النص للاجابة عن هذا السؤال."
4. حافظ على اللهجة او اللغة المناسبة لسؤال المستخدم.
5. اجعل الاجابة مختصرة وواضحة.
6. اذكر اسم الحلقة المصدر في نهاية الاجابة.

السياق المسترجع:
{context}

سؤال المستخدم:
{question}

الاجابة:
""".strip()