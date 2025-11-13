import pandas as pd
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

df = pd.read_csv('/dshome/ddualab/jiwon/ConVIRT-pytorch/data/fitz_df.csv')
df_short = df[:4]

def generate_sentence(fitzpatrick_scale, diameter_1, diameter_2, itch, grew, hurt, changed, bleed, elevation):
    prompt = """    
    You are a dermatologist, treating various patients from Fitzpatrick scale 1 to 6. 
    You usually write your findings by looking at the condition of the lesion to diagnose skin cancer positive or malignant.
    """
    model = genai.GenerativeModel(model_name="gemini-1.5-flash", system_instruction=prompt)

    response = model.generate_content([f"""Write at least 4 senetences for detailed skin characteristics of the skin color and at least 4 sentences for symtoms of lesion. 
                                       The Fitzpatrick type is {fitzpatrick_scale}. Explain the skin characteristics according to the skin color.
                                       Diameter of the lesion is {diameter_1}mm for horizontal, {diameter_2} for vertical, and if either horizontal or vertical is more than 6mm, it is larger than normal.
                                       Symptoms of the lesion as follows. itch : {itch}, grew : {grew}, hurt : {hurt}, changed : {changed}, bleed : {bleed}, elevation : {elevation}
                                       """])
    return response.text

df_short['generated_sentence'] = df_short.apply(lambda x: generate_sentence(x['fitzpatrick_scale'], x['diameter_1'], x['diameter_2'], x['itch'], x['grew'], x['hurt'], x['changed'], x['bleed'], x['elevation']), axis=1)

print(df_short.to_string())
