import pandas as pd
from openai import OpenAI
import os
import base64
import requests

def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# image_path = '/dshome/ddualab/jiwon/skin_cancer_fairness/data/skin_cancer_images/PAT_441_2868_663.png'
# base64_image = encode_image(image_path)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-proj-G1BLQt4pDrBeNM3AR2swT3BlbkFJJsDgCsEwiuNzTI6sGdlr"))

df = pd.read_csv('/dshome/ddualab/jiwon/skin_cancer_fairness/data/fitz_df.csv')

def generate_sentence(fitzpatrick_scale, diameter_1, diameter_2, itch, grew, hurt, changed, bleed, elevation, img_id, label):
    image_path = '/dshome/ddualab/jiwon/skin_cancer_fairness/data/skin_cancer_images/' + img_id 
    base64_image = encode_image(image_path)
    
    response = client.chat.completions.create(
        model = 'gpt-4o',
        messages=[
            {
                "role": "system", 
                "content": "You are a helpful dermatologist treating various patients from Fitzpatrick scale 1 to 6. You usually write your findings by looking at the condition of the lesion to diagnose skin cancer positive or malignant."
                }, 
                    
            {
                "role": "user", 
                "content": [
                    {
                    "type":"text",
                    "text":f"""Write at least 4 sentences for symtoms of lesion and at least 4 senetences for characteristics of the skin color. 
                                Explain the characteristics according to the image and the information below. 
                                The Fitzpatrick type is {fitzpatrick_scale}. 
                                The label of this case is {label}, where ACK, NEV, SEK represents benign whereas BCC, MEL, SCC represents malignant.
                                Symptoms of the lesion as follows. itch : {itch}, grew : {grew}, hurt : {hurt}, changed : {changed}, bleed : {bleed}, elevation : {elevation}
                                As for diameter of the lesion, {diameter_1}mm for horizontal, {diameter_2}mm for vertical. If either horizontal or vertical is more than 6mm, it is larger than normal.
                                Instead of dividing the skin color and lesion description into parts, write them down naturally within one paragraph. 
                                Write down the actual condition of the patient as if you're looking at it yourself and making your opinion.
                                If it is malignant, emphasizes the seriousness and if it is positive, write it down as a nuance that reassures the patient.
                                """
                    }, 
                    {
                    "type":"image_url",
                    "image_url":{
                        "url":f"data:image/png;base64,{base64_image}"
                    }
                    }
                ]
            }
        ]
    )

    sentence = response.choices[0].message.content
    return sentence

df['generated_sentence'] = df.apply(lambda x: generate_sentence(x['fitzpatrick_scale'], x['diameter_1'], x['diameter_2'], x['itch'], x['grew'], x['hurt'], x['changed'], x['bleed'], x['elevation'], x['img_id'], x['label']), axis=1)

# print(df.to_string())
df.to_csv('data/vlm_generated_df.csv', index=False)