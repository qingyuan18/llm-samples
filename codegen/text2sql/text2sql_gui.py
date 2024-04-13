## text2sql playground 20231024 上线版本  #####

import streamlit as st
from func import *
from typing import Dict
from typing import Any, Dict, List, Optional

aos_client = OpenSearch(
            hosts=[{'host': aos_endpoint, 'port': 443}],
            http_auth = pwdauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection)

aos_index="prompt-optimal-index"



db = SQLDatabase.from_uri(
    "mysql+pymysql://admin:admin12345678@"+dburi+"/llm",
    sample_rows_in_table_info=0)


## logic function
def gen_sql(user_input:str):
    global db
    if db is None:
       db = SQLDatabase.from_uri(
            "mysql+pymysql://admin:admin12345678@"+dburi+"/llm",
            sample_rows_in_table_info=0,
            include_tables=st.session_state['table_name'])

    db_chain = CustomerizedSQLDatabaseChain.from_llm(bedrock_llm, db, verbose=True, return_sql=True,return_intermediate_steps=False)
    response=db_chain.run(user_input)
    return response

def get_exact_prompt(query:str):
    ##input embedding
    query_embedding = get_vector_by_sm_endpoint(query, sm_client, embedding_endpoint_name)
    ##语义检索获取相似prompt
    similiary_prompts = aos_knn_search_v2(aos_client, "exactly_query_embedding",query_embedding[0], aos_index, size=10)
    ##返回最相似prompt string
    if len(similiary_prompts)>1:
       return similiary_prompts[0]["table_name"].strip(),similiary_prompts[0]["exactly_query_text"].strip()
    else:
       return None


# 确认按钮（映射到字典）
def get_table_mapping():
    selected_values = [value_mapping[option] for option in selected_options]
    st.session_state['table_name']=selected_values
    st.success("查询库表设置为"+','.join(selected_values))

# 字典映射
value_mapping = {
    "派车单明细": "ads_bi_quality_monitor_shipping_detail",
    "货主画像统计表": "ads_customer_portrait_index_sum_da",
    "车辆画像表": "ads_truck_portrait_index_sum_da",
    "客户企业站点基础信息表":"dim_customer_enterprise_station_base_info",
    "高德行政区域表":"dim_gaode_city_info_v2",
    "车辆基本属性纬度表":"dim_pub_truck_info",
    "车辆合作关系主表":"dim_pub_truck_tenant",
    "根节点订单对应运单信息":"dws_ots_waybill_info_da",
    "站点画像指标表":"dws_station_portrait_index_sum_da",
    "车辆画像指标表":"dws_truck_portrait_index_sum_da"
}
    
    
##session 保存库表信息, prompt修订
if 'table_name' not in st.session_state:
    st.session_state['table_name']=[]
if 'exactly_prompt' not in st.session_state:
    st.session_state['exactly_prompt']=""
    
###############################左侧菜单栏#################
st.sidebar.title("库表设置")
placeholder = st.empty()
# 多选下拉菜单
selected_options = st.sidebar.multiselect("查询库表", ["派车单明细", "货主画像统计表", "车辆画像表","客户企业站点基础信息表",
                                                      "高德行政区域表","车辆基本属性纬度表","车辆合作关系主表",
                                                      "根节点订单对应运单信息","站点画像指标表","车辆画像指标表"], key="multi_select")

# 确认按钮（左侧）
if st.sidebar.button("确认查询库表"):
    get_table_mapping()


############################右侧内容区域###################
st.title("sql生成")
# 文本输入框
prompt_container = st.empty()
user_input=prompt_container.text_input("查询库表",max_chars=100,value=st.session_state['exactly_prompt'])
#user_input = st.text_input("你要查询什么？", "")

# 单选框
show_message = st.checkbox("精准查询优化")

# 确认按钮（右侧）
if st.button("生成sql"):
    if show_message:
        extact_tables,extact_prompt = get_exact_prompt(user_input)
        if extact_tables is not None:
            st.session_state['table_name'] = extact_tables.split(",")
        if extact_prompt is not None:
           st.write("你是要查询类似内容么？"+extact_prompt)
           #if st.button("复制并修改"):
           #   st.session_state['exactly_prompt']=extact_prompt
           #   prompt_container.empty()
           #   prompt_container.text_input("查询库表",max_chars=100,value=st.session_state['exactly_prompt'])
    result_sql = gen_sql(user_input)
    placeholder.write(result_sql)




