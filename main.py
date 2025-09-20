import pandas as pd 
import os 
import json
import math
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from typing import TypedDict
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
load_dotenv()

path = "data_mf_with_email - Sheet1.csv"
df = pd.read_csv(path)

df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(' ', '')

class AgentState(TypedDict):
    customer_data: dict
    profile: dict
    channel: str
    
class ProfilingResult(BaseModel):
    segment: str = Field(description="Segmen risiko nasabah: [Risiko rendah, Risiko Menengah, Resiko Tinggi]")
    reason: str = Field(description="Alasan singkat penentuan segmen")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key = os.environ["GEMINI_API_KEY"], model_kwargs = {"seed":42}, temperature=0.2)

parser = JsonOutputParser(pydantic_object=ProfilingResult)

profiling_prompt = PromptTemplate(
    template="""
    Anda adalah seorang analis risiko kredit senior. Berdasarkan data nasabah berikut, tentukan segmen risikonya.

    Data Finansial & Perilaku Nasabah:
    - Angsuran Tertunggak: {AngsuranTertunggak}
    - Total Denda: {TotalDenda}
    - Angsuran per bulan: {Angsuranperbulan}
    - Progress Pinjaman: Angsuran ke-{AngsuranKe} dari total {Tenor} bulan.
    - Perkiraan Hari Keterlambatan (ODH): {ODH_hari} hari
    - Pekerjaan: {PekerjaanKonsumen}

    Tugas Anda:
    1. Fokus pada 'Angsuran Tertunggak' dan 'Total Denda' sebagai indikator utama.
    2. Pertimbangkan progress pinjaman dan perkiraan hari keterlambatan (ODH).
    3. Tentukan segmen nasabah: [Risiko Rendah, Risiko Menengah, Risiko Tinggi].
    4. Berikan alasan singkat berdasarkan data yang paling berpengaruh.

    {format_instruction}
    """,
    input_variables=["AngsuranTertunggak", "TotalDenda", "Angsuranperbulan", "AngsuranKe", "Tenor", "ODH_hari", "PekerjaanKonsumen"],
    partial_variables={"format_instruction" : parser.get_format_instructions()}
)

profiling_chain = profiling_prompt | llm | parser

def convert_str_to_int(x):
    clean_text = re.sub(r'\D', '', x)
    return clean_text

def rule_based_profiling(customer):
    odh = customer.get('ODH_hari', 0)
    sisa_tenor = customer.get('SisaTenor', 99)
    angsuran_tertunggak = int(convert_str_to_int(customer.get('AngsuranTertunggak', 0)))
    angsuran_perbulan = int(convert_str_to_int(customer.get('Angsuranperbulan', 1)))
    
    bulan_menunggak = angsuran_tertunggak / angsuran_perbulan if angsuran_perbulan > 0 else 0
    
    if bulan_menunggak > 3 or odh > 90:
        return {"segment" : "Risiko Tinggi", "reason" : "ODH Sangat Tinggi (>90 Hari) dan memiliki tunggakan lebih dari 3 bulan"}
    if bulan_menunggak < 3 and sisa_tenor < 3:
        return {"segment" : "Risiko Rendah", "reason" : "Akan segera lunas"}
    
def profiling_node(state: AgentState)->AgentState:
    """Node untuk melakukan profiling customer"""
    customer = state['customer_data']
    
    profile = rule_based_profiling(customer)
    
    if profile is None: 
        try: 
            print("Case di handel oleh LLM")
            profile = profiling_chain.invoke(customer)
        except Exception as e:
            profile = {"segment" : "Error", "reason" : f"Gagal memproses dengan LLM: {e}"}
    else:
        print("Case di handle Rule Based")
    
    return {"profile":profile}
    
def channel_selection_node(state: AgentState) -> AgentState:
    profile  = state['profile']
    customer = state['customer_data']
    channel = "WhatsApp"
    pekerjaan = customer.get('PekerjaanKonsumen', ' ').lower()
    email_valid = customer.get('Email') and customer['Email'] != 'NULL'
    
    if profile['segment'] in ['Risiko Rendah', 'Risiko Menengah']:
        if (pekerjaan in ['karyawan', 'professional&pns']) and email_valid:
            channel = "Email"
        else:
            channel = "WhatsApp"
            
    return {"channel": channel}

graph = StateGraph(AgentState)
graph.add_node('profiling', profiling_node)
graph.add_node('channel', channel_selection_node)
graph.set_entry_point('profiling')
graph.add_edge('profiling', 'channel')
graph.add_edge("channel", END)
app = graph.compile()

df['ODH_hari'] = df['ODH'].apply(lambda x: math.ceil(int(convert_str_to_int(x))/24))

result_list = []
    
for idx, row in df.head(5).iterrows():
    customer_data = row.to_dict()
    print(f"memproses calon customer: {idx + 1}: {customer_data['NamaKonsumen']}")        
    inputs = {"customer_data" : customer_data}
    final_state = app.invoke(inputs)
    
    final_data = {
        **customer_data, 
        'predicted_segment': final_state.get('profile', {}).get('segment', 'N/A'),
        'reasoning' : final_state.get('profile', {}).get('reason', 'N/A'),
        'recommended_channel': final_state.get('channel', 'N/A')
    }
    result_list.append(final_data)
    
result_df = pd.DataFrame(result_list)

print(result_df[[
    'NamaKonsumen', 
    'PekerjaanKonsumen', 
    'AngsuranTertunggak',
    'ODH_hari',
    'predicted_segment',
    'reasoning',
    'recommended_channel'
]])
