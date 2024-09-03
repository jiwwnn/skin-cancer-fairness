import pandas as pd
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-proj-G1BLQt4pDrBeNM3AR2swT3BlbkFJJsDgCsEwiuNzTI6sGdlr"))

df = pd.read_csv('/dshome/ddualab/jiwon/skin_cancer_fairness/data/vlm_generated_df.csv')
df_short = df[:4]

# def sentence_cls(text):    
#     response = client.chat.completions.create(
#         model='gpt-3.5-turbo',
#         messages=[
#             {
#                 "role": "system", 
#                 "content": "You are a helpful dermatologist treating various patients from Fitzpatrick scale 1 to 6. You usually write your findings by looking at the condition of the lesion to diagnose skin cancer positive or malignant."
#              }, 
                 
#             {
#                 "role": "user", 
#                 "content": f"""
#                             Please classify the following text into two categories: 
#                             1. Information about the lesion.
#                             2. Information about the skin color.
                            
#                             Text: "{text}"
                            
#                             Provide your response in the format:
#                             1. Lesion information: ...
#                             2. Skin color information: ...
#                             """
#             }
#         ]
#     )
#     output = response.choices[0].message.content
    
#     lesion_description = output.split('1. Lesion information: ')[1].split('2. Skin color information: ')[0].strip()
#     fitzpatrick_description = output.split('2. Skin color information: ')[1].strip()
    
#     return lesion_description, fitzpatrick_description

# df[['lesion_description', 'fitzpatrick_description']] = df.apply(lambda x: pd.Series(sentence_cls(x['generated_sentence'])), axis=1)

# # print(df.to_string())
# df.to_csv('data/sep_vlm_generated_df.csv', index=False)

def fitzpatrick_cls(text):    
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {
                "role": "system", 
                "content": "You are a helpful dermatologist treating various patients from Fitzpatrick scale 1 to 6. You usually write your findings by looking at the condition of the lesion to diagnose skin cancer positive or malignant."
             }, 
                 
            {
                "role": "user", 
                "content": f"Sort out sentences about the skin color from the given {text}."
            }
        ]
    )
    fitz_sentence = response.choices[0].message.content
    return fitz_sentence

def lesion_cls(text):    
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {
                "role": "system", 
                "content": "You are a helpful dermatologist treating various patients from Fitzpatrick scale 1 to 6. You usually write your findings by looking at the condition of the lesion to diagnose skin cancer positive or malignant."
             }, 
                 
            {
                "role": "user", 
                "content": f"Sort out sentences about the lesion condition from the given {text}."
            }
        ]
    )
    lesion_sentence = response.choices[0].message.content
    return lesion_sentence

df_short.loc[:, 'fitzpatrick_description'] = df_short.apply(lambda x: fitzpatrick_cls(x['generated_sentence']), axis=1)
df_short.loc[:, 'lesion_description'] = df_short.apply(lambda x: lesion_cls(x['generated_sentence']), axis=1)

print(df_short.to_string())
df_short.to_csv('data/sep_vlm_generated_df.csv', index=False)