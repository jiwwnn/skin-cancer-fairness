import pandas as pd
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-proj-G1BLQt4pDrBeNM3AR2swT3BlbkFJJsDgCsEwiuNzTI6sGdlr"))

df = pd.read_csv('/dshome/ddualab/jiwon/skin_cancer_fairness/data/fitz_df.csv')
df_short = df[:4]

def generate_sentence(fitzpatrick_scale, diameter_1, diameter_2, itch, grew, hurt, changed, bleed, elevation):    
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {
                "role": "system", 
                "content": "You are a helpful dermatologist treating various patients from Fitzpatrick scale 1 to 6. You usually write your findings by looking at the condition of the lesion to diagnose skin cancer positive or malignant."
             }, 
                 
            {
                "role": "user", 
                "content": f"""Write at least 4 senetences for skin characteristics of the skin color and at least 4 sentences for symtoms of lesion. 
                            The Fitzpatrick type is {fitzpatrick_scale}. Explain the skin characteristics according to the skin color.
                            Symptoms of the lesion as follows. itch : {itch}, grew : {grew}, hurt : {hurt}, changed : {changed}, bleed : {bleed}, elevation : {elevation}
                            As for diameter of the lesion, {diameter_1}mm for horizontal, {diameter_2} for vertical. If either horizontal or vertical is more than 6mm, it is larger than normal.
                            """
            }
        ]
    )
    sentence = response.choices[0].message.content
    return sentence

df_short['generated_sentence'] = df_short.apply(lambda x: generate_sentence(x['fitzpatrick_scale'], x['diameter_1'], x['diameter_2'], x['itch'], x['grew'], x['hurt'], x['changed'], x['bleed'], x['elevation']), axis=1)


print(df_short.to_string())
