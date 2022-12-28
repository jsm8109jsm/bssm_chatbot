import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

@st.cache(allow_output_mutation=True)
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache(allow_output_mutation=True)
def get_dataset():
    df = pd.read_csv('chatbot.csv')
    df['embedding'] = df['embedding'].apply(json.loads)
    return df

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

model = cached_model()
df = get_dataset()

st.header('부산소프트웨어마이스터고 챗봇')
st.subheader("안녕하세요 소마고 챗봇입니다.")

tab1, tab2, tab3 = st.tabs(["학교소개", "입학안내", "문의"])
with tab1:
    st.header("저희 소마고를 소개합니다.")

    st.markdown("""4차 산업혁명 핵심기술의 근간은 ‘SW’로 글로벌 시장은 SW인재 중심으로 급속히 재편 중이며, 기업에서도 SW인재 개발 분야를 지원, 학력 불문한 SW영재양성을 필요로 하고 있습니다.<br><br><br>
부산소프트웨어마이스터고등학교에서는 모든 산업의 근간이 되는 소프트웨어 학습을 통해 사람과 생활에 유익한 소프트웨어를 개발하도록 교육과정을 운영하고 소통Communication, 배려Consideration, 협업Cooperation, 창의성Creativity으로 기계적 시스템을 이용해 새로운 가치를 만들어 내고 이를 공유하여 세상을 이롭게 하는 하이브리드형 인재를 양성하여 SW 분야의 창의·융합형 영 마이스터로 성장하도록 적극 지원할 것입니다.<br><br><br>
부산소프트웨어마이스터고등학교는 교육공동체가 함께 만들어가는 교육과정 운영으로 더불어 행복하고 나누어 즐거운 학교로 대한민국을 넘어 글로벌 인재 양성의 메카로 우뚝 설 수 있도록 교육공동체와 함께 마이스터고의 역사를 써 내려가도록 노력하겠습니다.

""", unsafe_allow_html=True)
with tab2:
    st.header("입학 안내")

    st.markdown("""<a href="https://school.busanedu.net/bssm-h/na/ntt/selectNttList.do?mi=1019680&bbsId=5095434">입학안내 바로가기</a>""", unsafe_allow_html=True)
with tab3:
    st.header("챗봇에게 무엇이든 물어보세요!")

    with st.form('form', clear_on_submit=True):
        user_input = st.text_input('사용자 : ','')
        submitted = st.form_submit_button('전송')

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []

    if submitted and user_input:
        embedding = model.encode(user_input)

        df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
        answer = df.loc[df['distance'].idxmax()]

        st.session_state.past.append(user_input)
        if answer['distance'] > 0.5:
            st.session_state.generated.append(answer['챗봇'])
        else:
            st.session_state.generated.append("무슨 말인지 모르겠어요. 여기다가 물어보세요(051-971-2153)")
        

    for i in range(len(st.session_state['past'])):
        st.markdown(
            """
            <div class="root right">
                <div class="chat">{0}</div>
                <div class="circle">유</div>
            </div>
            <div class="root left">
                <div class="circle">봇</div>
                <div class="chat">{1}</div>
            </div>
            """.format(st.session_state['past'][i], st.session_state['generated'][i]), unsafe_allow_html=True
        )

st.markdown("""
            <footer class="footer">
                학교 연락처 : 051-971-2153<br>
                학교 홈페이지 : <a href="https://school.busanedu.net/bssm-h/main.do">https://school.busanedu.net/bssm-h/main.do</a>
            </footer>
        """, unsafe_allow_html=True)


