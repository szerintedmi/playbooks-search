"""
Search either with SentenceTransformers or OpenAI
TODO: it became a mess with too much logic intertwined with the UI
TODO: debug url param?
"""
import streamlit as st
import os
import pandas as pd
import logger
import time

# import search_SentenceTransformers as search_module
import search_oai as search_module
import openAI_Utils

def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == os.getenv("APP_PASSWORD"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" in st.session_state and st.session_state["password_correct"] == True:
        return True

    st.info("Ask EE Playbooks\n\n"
        "Experimental, proof-of-concept tool built on OpenAI language models.\n\n"
        "🔐 The app is password-protected. Please enter the password.")

    st.text_input("Password", type="password", on_change=password_entered, key="password", 
                    placeholder="Ask on #ask-playbooks EE slack channel for password")

    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 Password incorrect")

    return False

def get_model_params():
    match st.session_state.gpt_model:
        case "gpt-3.5-turbo": 
            return { 
                "TOKEN_PRICE": 0.002,
                "MODEL_MAX_CONTEXT_LEN": 4096,
                "PROMPT_CONTENT_MAX_TOKENS": 2700 }

        case "gpt-4":
            return { 
                "TOKEN_PRICE": 0.03,
                "MODEL_MAX_CONTEXT_LEN": 8192,
                "PROMPT_CONTENT_MAX_TOKENS": 5400 }

        case _: raise NotImplementedError(f"Selected GPT_MODEL {st.session_state.gpt_model} is not implemented")


if check_password():
    if "gpt_model" not in st.session_state:
        st.session_state.gpt_model = "gpt-3.5-turbo"

    # not including the fix part of the prompt + added title for each section
    CONTENT_PREVIEW_LEN = 500
    NEWLINE = "\n"  # workaround for \ not allowed in f strings

    st.set_page_config(layout="wide", page_title="Ask EE")

    st.write("# Ask EE Playbooks")
    col1, col2 = st.columns(2)
    with col1:
        st.info(
            "Currently indexed playbooks:\n\n"
            "Advice Process, Chaos Day, Digital Platform, Inception, Remote Working, "
            "Secure Delivery and You Build It You Run It (YBIYRI)\n\n"
            "Experimental proof of concept. For accurate information, always refer to the official [EE Playbooks](https://www.playbook.ee/).")  

    with col2:
        # st.info(
        #     "The questions are logged (without user information) to evaluate the engine's performance.\n\n")
        st.radio("OpenAI model:", ["gpt-3.5-turbo", "gpt-4"], key="gpt_model", horizontal=False, help="gpt-3.5-turbo: perfectly good for most cases and much cheaper (ca. $0.005 per query)\n\ngpt-4: better in reasoning but expensive (ca. $0.10 per query)")

    query = st.text_input(label_visibility="collapsed", label="Your question:",
                        placeholder="Enter your question here and press enter", value="")

    if query.strip() == "":
        st.write("""Ideas to try:

    * What is the difference between a product manager and a product owner?
    * What are your key recommendations for secure delivery?
    * How can I improve my release cycles?
    * What's the difference between an inception and a discovery?
    * How should I start a digital platform?
    * How to run an inception?
    * How to run a discovery?
    """)

    all_result = []
    question_prompt = ""

    selected_content = []
    prompt_token_count = 0


    if query.strip() != "":
        with st.spinner('Analyzing question ...'):
            all_result = search_module.search(query)

            if len(all_result) > 0:
                all_result = pd.DataFrame(all_result)
                all_result = all_result.merge(
                    search_module.get_corpus(), left_on='corpus_id', right_index=True)

                selected_result = []

                for index, row in all_result.iterrows():
                    # Checks if result should be added to selected_result + avoid duplicate content

                    if row['score'] < search_module.RESULT_MIN_SCORE:
                        break

                    if row['tokensLength'] + prompt_token_count <= get_model_params()['PROMPT_CONTENT_MAX_TOKENS']:
                        add_row = True
                        row_path_comparator = "#".join(
                            [""] + row.navInfo["sectionPath"] + [""])

                        for x_idx, x in enumerate(selected_result):
                            # If part of this section is already in selected_result as subsection
                            #   then remove the subsection before adding this section. E.g.:
                            #       row already in selected_result (x):   title1 -> title2 -> title3  ===> remove
                            #                              current row:   title1 -> title2            ===> add
                            x_path_comparator = "#".join(
                                [""] + x.navInfo["sectionPath"] + [""])

                            if x['sourcePath'] == row['sourcePath'] and row_path_comparator in x_path_comparator:
                                selected_result.pop(x_idx)

                            # if current section (row) is already in selected_result (x) as part of a parent section then skip
                            if x['sourcePath'] == row['sourcePath'] and x_path_comparator in row_path_comparator:
                                add_row = False
                                break

                        if add_row:
                            selected_result.append(row)

                        prompt_token_count = sum(d['tokensLength']
                                        for d in selected_result)

                for row in selected_result:
                    # create context for the answer based on selected_result
                    # TODO: use list comprehension
                    selected_content.append(
                        f"{row.fullTitle}\n\n{row['content']}")

                question_prompt = f"""\
        Use the below sections from EE PLAYBOOKs in markdown format to answer the subsequent question.
        Give very detailed answer and use bullet points.
        Answer the question as truthfully as possible and
        if you're unsure of the answer, say "Sorry, I don't know"

        PLAYBOOK SECTIONS:
        {(NEWLINE + NEWLINE).join(selected_content)}

        QUESTION: {query}
        """


    if len(selected_content) > 0:
        response_container = st.empty()
        with st.spinner('Firing neurons ...'):
            max_completition_tokens = int(min([prompt_token_count * 0.75, get_model_params()['MODEL_MAX_CONTEXT_LEN'] - prompt_token_count]))

            prompt_messages = [
                    {'role': 'system', 'content': 'You answer questions about EE Playbooks.'},
                    {'role': 'user', 'content': question_prompt},
                    ]

            query_params = {
                "messages": prompt_messages,
                "model": st.session_state.gpt_model,
                "max_tokens": max_completition_tokens,
                "temperature":0,
                "stream":True
            }

            start_time = time.time()
            
            completion_resp = openAI_Utils.get_completition(query_params)

            response_token_ct = 0
            response_content = ""

            # iterate through the stream of events
            for chunk in completion_resp:
                response_token_ct += 1
                response_delta = chunk['choices'][0]['delta']
                try:
                    response_content += response_delta.content
                    response_container.write(response_content)
                except AttributeError:
                    pass # first chunk doesn't content, only role attribute

            end_time = time.time()

            # prompt_response = completion_resp.choices[0].message.content
            # st.write(prompt_response)

            prompt_token_ct = openAI_Utils.num_tokens_from_messages(prompt_messages, st.session_state.gpt_model)
            total_token_usage = prompt_token_count + response_token_ct 

            logger.log_search(query, search_module.MODEL_NAME, all_result,
                            query_params, response_content, total_token_usage, end_time - start_time)


            st.write(
                f"`{st.session_state.gpt_model} tokens used: {total_token_usage} (${ round(total_token_usage * get_model_params()['TOKEN_PRICE'] / 1000, 4)}) | prompt_token_ct: {prompt_token_ct} | response_token_ct: {response_token_ct} | prompt tokens: {prompt_token_count} | max_completition_tokens: {max_completition_tokens} | completion time: {round(end_time - start_time, 2)} `")

            st.write("## Sources")
            for row in selected_result:

                with st.expander(row.fullTitle):
                    st.write(
                        row.content[:CONTENT_PREVIEW_LEN] +
                        f"{'...' if len(row.content) > CONTENT_PREVIEW_LEN else '' }\n\n"
                        f"[Read all in the playbook >>]({row.navInfo['playbookUrl']})")

                    # Debug:
                    st.write(

                        f"`id: {row['corpus_id']} | path_depth: {row.navInfo['pathDepth']} | level: {row.level}"
                        f" | tokens: {row.tokensLength} | score: {round(row.score,4)}\n"
                        f" | {row.navInfo['playbookUrl']} | {row.sourcePath} | page_title: '{row.navInfo['pageTitle']}'"
                        f" | subtitles: '{row.navInfo['subTitles']}' | anchor_slug: '{row.navInfo['anchorSlug']}'`")

    elif query.strip() != "":
        st.warning(
            "Sorry, I was unable to find an answer to your question in the EE Playbooks that I am aware of.\n\n"
            "  Please note that I am an experimental tool, so it is possible that the answer is there and I simply failed to find it.\n\n"
            "  You may want to try rephrasing your question or checking the [EE Playbooks](https://www.playbook.ee/) for a possible answer.", icon="🤔")

        # unsafe_allow_html=True)


    # Display debug info
    if len(all_result) > 0:
        st.markdown(
            """<hr style="height:6px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

        with st.expander("Geeky debug info"):

            st.write("**All Results**")
            if len(all_result) > 0:

                st.dataframe(
                    all_result[["corpus_id", 'score', 'level', 'navInfo', 'content', 'tokensLength']], height=200)

            st.write("**Full Corpus**")
            st.write(search_module.get_corpus())

            st.write("**Prompt**")
            st.write(
                f"Prompt tokens length: {prompt_token_count} | From {len(selected_content)} sections")
            st.code(question_prompt, language="markdown")
