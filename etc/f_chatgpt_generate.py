import pandas as pd
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-proj-G1BLQt4pDrBeNM3AR2swT3BlbkFJJsDgCsEwiuNzTI6sGdlr"))

df = pd.read_csv('/dshome/ddualab/jiwon/ConVIRT-pytorch/data/fitz_df.csv')

def fitzpatrick_txt_gen(fitzpatrick_scale):    
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {
                "role": "system", 
                "content": "You are a helpful dermatologist treating various patients from Fitzpatrick scale 1 to 6. You usually write your findings by looking at the condition of the lesion to diagnose skin cancer positive or malignant."
             }, 
                 
            {
                "role": "user", 
                "content": f"""Write at least 4 senetences for skin characteristics of the skin color. 
                            The Fitzpatrick type is {fitzpatrick_scale}. Explain the skin characteristics according to the skin color.
                            """
            }
        ]
    )
    sentence = response.choices[0].message.content
    return sentence

def lesion_txt_gen(diameter_1, diameter_2, itch, grew, hurt, changed, bleed, elevation):    
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {
                "role": "system", 
                "content": "You are a helpful dermatologist treating various patients from Fitzpatrick scale 1 to 6. You usually write your findings by looking at the condition of the lesion to diagnose skin cancer positive or malignant."
             }, 
                 
            {
                "role": "user", 
                "content": f""""Write at least 4 sentences for symtoms of lesion.
                            Symptoms of the lesion as follows. itch : {itch}, grew : {grew}, hurt : {hurt}, changed : {changed}, bleed : {bleed}, elevation : {elevation}
                            As for diameter of the lesion, {diameter_1}mm for horizontal, {diameter_2} for vertical. If either horizontal or vertical is more than 6mm, it is larger than normal.
                            """
            }
        ]
    )
    sentence = response.choices[0].message.content
    return sentence

df['fitzpatrick_description'] = df.apply(lambda x: fitzpatrick_txt_gen(x['fitzpatrick_scale']), axis=1)
df['lesion_description'] = df.apply(lambda x: lesion_txt_gen(x['diameter_1'], x['diameter_2'], x['itch'], x['grew'], x['hurt'], x['changed'], x['bleed'], x['elevation']), axis=1)

df.to_csv('description_df.csv', index=False)
