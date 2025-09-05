from datetime import datetime


def questions_and_searches_prompt(
    fact_checking_topic: str, lang: str = "es", location: str = None
) -> str:
    return f"""
You are tasked with generating a counter-narrative for a given statement. The statement can also be formulated as a question. This involves creating a plan, generating critical questions, and formulating Google searches to gather information that may challenge or verify the statement.

Here is the statement to analyze:
{fact_checking_topic}


First, create a plan to approach this task. Your plan should outline the steps you'll take to gather information and analyze the statement. You should gather information and compare relevant data.

Next, generate a set of critical questions that would help gather information about the topic. These questions should be critical and aimed at uncovering various aspects of the statement. You can generate a maximum of 6 questions, so be concise and specific. Questions should start with "¿" and end with "?".

Finally, generate a set of Google searches that would help retrieve the necessary information to evaluate the statement as well as the questions you generated. You can generate a maximum of 6 searches, so be concise and specific.

Present your response in the following JSON format. The JSON should include three main keys: "plan", "questions", and "searches". "plan" should be a string in Markdown formatting. "questions" and "searches" should be arrays containing the items you generated in each step. The current year is {datetime.now().year}. 

Your answer should be in {'Spanish' if lang == 'es' else 'English'}.

    """.strip()


def qa_answer_prompt(question: str, data_block: str, lang: str = "es") -> str:
    return f"""
You will be given a series of text chunks and a question. Your task is to answer the question based on the information provided in the text chunks, using appropriate citations.
First, you will receive the text chunks in the following format:
{data_block}

You should answer the following question based on the information in the text chunks:
{question}

To complete this task, follow these steps:

1. Carefully read and analyze all the text chunks provided.
2. Identify the chunks that contain information relevant to answering the question.
3. Formulate an answer to the question using the information from the relevant chunks.
4. When using information from a specific chunk, cite it using the format [X], where X is the chunk number.
5. If multiple chunks support a statement, include all relevant citations, e.g., [1] [3] [5].
6. Ensure that your answer is in the same language as the question.
7. Present your answer in a clear and concise manner, using complete sentences.
8. The user can only see the question and the answer, not the text chunks. Therefore, make sure your answer is self-contained and provides all the necessary information. Do not make any direct references to the text chunks in your answer.

Remember to use the information provided in the text chunks and cite your sources appropriately. Do not include any information that is not present in the given text chunks.

Your answer should be in {'Spanish' if lang == 'es' else 'English'}.
""".strip()


def article_prompt(
    fact_checking_topic, data_block, question_block, lang: str = "es"
) -> str:
    return f"""
You are tasked with generating a fact check for a given statement or question. You will be provided with critical questions and text chunks to assist you in this task. Follow these instructions carefully:

1. Here is the statement or question to be fact-checked:
{fact_checking_topic}

2. Here you have a set of critical questions to guide your fact check, you should answer them based on the information provided in the text chunks:
{question_block}

3. Here are the text chunks you should use as your source of information:
{data_block}

4. Generate a fact check based on the provided text chunks. 
    - Use the critical questions to guide your analysis. When referencing information from the text chunks, cite them using the format [1], [2], etc., corresponding to their ID numbers. 
    - Ensure that your fact check is based solely on the information provided in the text chunks. If you cannot find relevant information in the text chunks to address any part of the statement or critical questions, clearly state that there is insufficient information to make a determination on that specific point. 
    - Use as much detail as possible to support your fact check. 
    - You should find all the arguments in favor and against the statement or question. Your task is not to persuade but to factually present the information and different perspectives.

5. Structure your fact check in exactly three paragraphs. Paragraphs should only be separated by a blank line.
    - First paragraph: You must describe the arguments that support the statement or answer the question affirmatively. If you cannot find arguments in favor within the provided text fragments, you must explain that you have not found any.
    - Second paragraph: You must describe the arguments that refute the statement or answer the question negatively. If you cannot find arguments against within the provided text fragments, you must explain that you have not found any.
    - Third paragraph: You must explain your conclusion regarding the truthfulness of the statement or question. You must explain whether the statement or question is true, false, or if it cannot be determined.

6. In your citations and analysis, refer to the text chunks by their ID numbers (e.g., [1], [2]). Ensure that you only use information from the provided text chunks.

7. Conclude your fact check with a sentence that determines whether the statement is true, false, or if the truth cannot be determined based on the available information.

Very important: 
    - If you include any technical terms or concepts, make sure to explain them clearly.
    - If you include any acronyms, make sure to expand them the first time you use them.
    - The user cannot see the text chunk or the question. Therefore, make sure your answer is self-contained and provides all the necessary information.
    - Do make explicit references to the text chunks in your answer, the user will not have access to them. So do not say "As mentioned in the first text chunks provided..." but rather "There is evidence that...".
    - Your paragraphs can be as long as necessary to provide a comprehensive fact check, do not worry about the length.
    - If no sopporting or contradicting evidence is found, state that clearly. If not enough evidence is found the answer the questions above, state that clearly.

Your answer should be in {'Spanish' if lang == 'es' else 'English'}.
""".strip()


def image_description_prompt(fact_checking: str, lang: str = "es") -> str:
    return f"""
Given the following article title, I want to generate a header image using an image-generating AI. Could you generate a suitable image description for the article?
The description should be short but concise. Respond only with the image description; do not add anything else to your response. Answer in English.
Article Title: {fact_checking}
""".strip()


def get_metadata_prompt(
    fact_cheking_title: str, fact_checking: str, lang: str = "es"
) -> str:
    return f"""
Given a fact-checking article, I want you to generate a JSON with the following information:
1. Article title: The article title corrected for spelling and grammar. Keep the original title if there are no errors.
2. List of categories: For example, News, Sports, Entertainment, Science, Technology, Environment, Automotive, Politics, etc... Maximum 3 categories.
3. Tag: If the conclusion is "Fake", "True", or "Undetermined", extract it from the last paragraph of the article.
4. Main claim: Based on the tag, rewrite the article headline. The headline should be a statement that includes the main topic of the article and the fact-checking conclusion.

Example:
{{
    "title": "Are electric cars more polluting than gasoline cars?",
    "categories": ["Automotive", "Environment", "Science"],
    "label": "Fake",
    "main_claim": "Electric cars are not more polluting than gasoline cars"
}}

Article title: {fact_cheking_title}
Article: {fact_checking}

Your answer should be in {'Spanish' if lang == 'es' else 'English'}.
    """.strip()
