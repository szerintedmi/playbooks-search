"""
Search either with SentenceTransformers or OpenAI
TODO: it became a mess with too much logic intertwined with the UI
TODO: debug url param?
"""
import streamlit as st
import pandas as pd
import logger
import time

# import search_SentenceTransformers as search_module
import search_oai as search_module

# not including the fix part of the prompt + added title for each section
PROMPT_CONTENT_MAX_TOKENS = 2000
CONTENT_PREVIEW_LEN = 500

NEWLINE = "\n"  # workaround for \ not allowed in f strings

st.set_page_config(layout="wide")

st.write("# Ask EE")
col1, col2 = st.columns(2)
with col1:
    st.info(
        "Ask anything covered in EE Playbooks.\n\nPlaybooks included at the moment:\n\n"
        "Advice Process, Chaos Day, Digital Platform, Inception, Remote Working, "
        "Secure Delivery and You Build It You Run It (YBIYRI)")

with col2:
    st.info(
        "It's an experimental, proof-of-concept tool for EE internal use.\n\n"
        "The answers are generated and **NOT official EE answers or opinions**. "
        "The answer may occasionally be completely off due to the nature of the underlying language model.\n\n"
        "For accurate information, always refer to the official [EE Playbooks](https://www.playbook.ee/).\n\n"
        "The questions are logged (without user information) to evaluate the engine's performance.\n\n")

query = st.text_input(label_visibility="collapsed", label="Your question:",
                      placeholder="Type your question here and press enter", value="")

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
token_count = 0


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

                if row['tokensLength'] + token_count <= PROMPT_CONTENT_MAX_TOKENS:
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

                    token_count = sum(d['tokensLength']
                                      for d in selected_result)

            for row in selected_result:
                # create context for the answer based on selected_result
                # TODO: use list comprehension
                selected_content.append(
                    f"{row.fullTitle}\n\n{row['content']}")

            question_prompt = f"""\
    Answer the QUESTION based on the following PLAYBOOKs in markdown format.
    Give very detailed answer and use bullet points.
    Answer the question as truthfully as possible and
    if you're unsure of the answer, say "Sorry, I don't know"
    QUESTION: {query}
    PLAYBOOKS:
    {(NEWLINE + NEWLINE).join(selected_content)}
    ANSWER:
    """


if len(selected_content) > 0:
    with st.spinner('Firing neurons ...'):
        max_tokens = int(min([token_count * 0.75, 4000 - token_count]))

        query_params = {"model": "text-davinci-003",
                        "prompt": question_prompt,
                        "max_tokens": max_tokens,
                        "temperature": 0.2}

        start_time = time.time()
        resp = search_module.get_completition(query_params)
        end_time = time.time()

        print("Results (after {:.3f} seconds):".format(end_time - start_time))

        prompt_response = resp.choices[0].text
        st.write(prompt_response)

        logger.log_search(query, search_module.MODEL_NAME, all_result,
                          query_params, prompt_response, resp.usage.total_tokens, end_time - start_time)

        st.write(
            f"`tokens used: {resp.usage.total_tokens} | prompt tokens: {token_count} | max_tokens: {max_tokens} | completion time: {round(end_time - start_time, 2)} `")

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

    with st.expander("Geeky stuff"):

        st.write("**All Results**")
        if len(all_result) > 0:

            st.dataframe(
                all_result[["corpus_id", 'score', 'level', 'navInfo', 'content', 'tokensLength']], height=200)

        st.write("**Full Corpus**")
        st.write(search_module.get_corpus())

        st.write("**Prompt**")
        st.write(
            f"Prompt tokens length: {token_count} | From {len(selected_content)} sections")
        st.code(question_prompt, language="markdown")
