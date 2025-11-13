import pandas as pd
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-proj-G1BLQt4pDrBeNM3AR2swT3BlbkFJJsDgCsEwiuNzTI6sGdlr"))


response = client.chat.completions.create(
    model='gpt-3.5-turbo',
    messages=[
        {
            "role": "system", 
            "content": "You are a text to SQL query expert. Please help to generate a SQL query to answer the question. Your response should ONLY be based on the given context and follow the response guidelines and format instructions."
            }, 
                
        {
            "role": "user", 
            "content": f"""
                        ### Some example pairs of questions and corresponding SQL queries are provided based on similar questions:
                        
                        ### In a military database, there are 1,500 soldiers. 30% of them are assigned to units with a commander whose blood type is 'B'. Of the remaining soldiers, 40% are affiliated with units where the organization is '여단' and the unit’s total number of soldiers (Num) is less than 700. Among the rest, 50% are affiliated with units storing supplies where TYPE_C is '안테나' and Storage is located in a unit whose commander resides in '서울'. The remaining soldiers are affiliated with units where the affiliated class requires a certificate named '통신자격증'. What percentage of the total soldiers are affiliated with units storing supplies where TYPE_C is '안테나'?
                        WITH TotalSoldiers AS (
                            SELECT 1500 AS total_soldiers
                        ),
                        SoldiersWithBloodTypeB AS (
                            SELECT total_soldiers * 0.3 AS num_soldiers_bloodTypeB
                            FROM TotalSoldiers
                        ),
                        RemainingSoldiersAfterBloodTypeB AS (
                            SELECT total_soldiers - num_soldiers_bloodTypeB AS remaining_after_bloodTypeB
                            FROM TotalSoldiers, SoldiersWithBloodTypeB
                        ),
                        SoldiersIn여단Under700 AS (
                            SELECT remaining_after_bloodTypeB * 0.4 AS num_soldiers_여단_under700
                            FROM RemainingSoldiersAfterBloodTypeB
                        ),
                        RemainingSoldiersAfter여단 AS (
                            SELECT remaining_after_bloodTypeB - num_soldiers_여단_under700 AS remaining_after_여단
                            FROM RemainingSoldiersAfterBloodTypeB, SoldiersIn여단Under700
                        ),
                        SoldiersIn안테나Supplies AS (
                            SELECT remaining_after_여단 * 0.5 AS num_soldiers_안테나_supplies
                            FROM RemainingSoldiersAfter여단
                        ),
                        RemainingSoldiersAfter안테나 AS (
                            SELECT remaining_after_여단 - num_soldiers_안테나_supplies AS remaining_after_안테나
                            FROM RemainingSoldiersAfter여단, SoldiersIn안테나Supplies
                        ),
                        FinalResult AS (
                            SELECT num_soldiers_안테나_supplies * 100.0 / total_soldiers AS percentage_soldiers_with_안테나_supplies
                            FROM TotalSoldiers, SoldiersIn안테나Supplies
                        )
                        SELECT percentage_soldiers_with_안테나_supplies
                        FROM FinalResult;

                        ### A total of 2,500 soldiers are recorded in the database. 20% of these soldiers are assigned to units with an organization type of '본부' and are located in 서울. Of the remaining soldiers, 60% are affiliated with units where the commander's position is '행정' and they have passed a physical examination with a Grade of 3. Of the remaining soldiers, 30% are affiliated with units storing supplies where TYPE_A is '물자' and the unit's Num exceeds 900. The rest are affiliated with units that require the ReqCert of '행정자격증'. What percentage of all soldiers are assigned to units requiring the ReqCert of '행정자격증'?
                        WITH TotalSoldiers AS (
                            SELECT 2500 AS total_soldiers
                        ),
                        SoldiersIn본부서울 AS (
                            SELECT total_soldiers * 0.2 AS num_soldiers_본부서울
                            FROM TotalSoldiers
                        ),
                        RemainingSoldiersAfter본부 AS (
                            SELECT total_soldiers - num_soldiers_본부서울 AS remaining_after_본부
                            FROM TotalSoldiers, SoldiersIn본부서울
                        ),
                        SoldiersWithPosition행정 AS (
                            SELECT remaining_after_본부 * 0.6 AS num_soldiers_position_행정
                            FROM RemainingSoldiersAfter본부
                        ),
                        RemainingSoldiersAfter행정 AS (
                            SELECT remaining_after_본부 - num_soldiers_position_행정 AS remaining_after_행정
                            FROM RemainingSoldiersAfter본부, SoldiersWithPosition행정
                        ),
                        SoldiersIn물자Above900 AS (
                            SELECT remaining_after_행정 * 0.3 AS num_soldiers_물자_above900
                            FROM RemainingSoldiersAfter행정
                        ),
                        RemainingSoldiersAfter물자 AS (
                            SELECT remaining_after_행정 - num_soldiers_물자_above900 AS remaining_after_물자
                            FROM RemainingSoldiersAfter행정, SoldiersIn물자Above900
                        ),
                        FinalResult AS (
                            SELECT remaining_after_물자 * 100.0 / total_soldiers AS percentage_soldiers_with_행정자격증
                            FROM TotalSoldiers, RemainingSoldiersAfter물자
                        )
                        SELECT percentage_soldiers_with_행정자격증
                        FROM FinalResult;

                        ### In a military database, there are 3,000 soldiers. 25% of them are assigned to units where the organization is '사단' and the unit’s latitude is greater than 37.0. Of the remaining soldiers, 50% are affiliated with units whose commander has blood type 'O' and has passed a physical examination with a Height greater than 170 cm and a Weight of less than 75 kg. Among the remaining soldiers, 40% are affiliated with units storing supplies where TYPE_B is '보급' and TYPE_A is '식량'. The rest are affiliated with units whose subordinates include '302' or '303'. What percentage of the total soldiers are affiliated with units storing supplies where TYPE_B is '보급'?
                        WITH TotalSoldiers AS (
                            SELECT 3000 AS total_soldiers
                        ),
                        SoldiersIn사단Above37 AS (
                            SELECT total_soldiers * 0.25 AS num_soldiers_사단_above37
                            FROM TotalSoldiers
                        ),
                        RemainingSoldiersAfter사단 AS (
                            SELECT total_soldiers - num_soldiers_사단_above37 AS remaining_after_사단
                            FROM TotalSoldiers, SoldiersIn사단Above37
                        ),
                        SoldiersWithBloodTypeO AS (
                            SELECT remaining_after_사단 * 0.5 AS num_soldiers_bloodTypeO
                            FROM RemainingSoldiersAfter사단
                        ),
                        RemainingSoldiersAfterBloodTypeO AS (
                            SELECT remaining_after_사단 - num_soldiers_bloodTypeO AS remaining_after_bloodTypeO
                            FROM RemainingSoldiersAfter사단, SoldiersWithBloodTypeO
                        ),
                        SoldiersIn보급Supplies AS (
                            SELECT remaining_after_bloodTypeO * 0.4 AS num_soldiers_보급_supplies
                            FROM RemainingSoldiersAfterBloodTypeO
                        ),
                        RemainingSoldiersAfter보급 AS (
                            SELECT remaining_after_bloodTypeO - num_soldiers_보급_supplies AS remaining_after_보급
                            FROM RemainingSoldiersAfterBloodTypeO, SoldiersIn보급Supplies
                        ),
                        SoldiersWithSubordinates AS (
                            SELECT remaining_after_보급 AS num_soldiers_with_subordinates
                            FROM RemainingSoldiersAfter보급
                        ),
                        FinalResult AS (
                            SELECT num_soldiers_보급_supplies * 100.0 / total_soldiers AS percentage_soldiers_with_보급_supplies
                            FROM TotalSoldiers, SoldiersIn보급Supplies
                        )
                        SELECT percentage_soldiers_with_보급_supplies
                        FROM FinalResult;
                        
                        ### Answer the question by SQLite SQL query only and with no explanation . You must minimize SQL execution time while ensuring correctness .
                        ### Sqlite SQL tables, with their properties:
                        #
                        # 인적정보 ( ID, Name, Address1, Address2, AcademicAbility, Certificate, Family, Rank, Position, Class, PE, Affiliation );
                        # 신체검사기록 ( ID, Date, Location, Height, Weight, Eye_L, Eye_R, BloodType, Grade );
                        # 병과 ( Class, ReqCert, ReqGrade );
                        # 자격증 ( ID, Name );
                        # 군수 ( ID, TYPE_A, TYPE_B, TYPE_C, Storage );
                        # 부대 ( ID, Longitude, Latitude, Commander, SubordinateUnit, Organization, Num );
                        #
                        ### Here is some data information about database references.
                        #
                        # 인적정보 ( ID [A001, A002, A003], Name [홍길동, 김철수, 이영희], Address1 [서울, 부산, 인천], Address2 [강남구, 해운대구, 남동구], AcademicAbility [대졸, 고졸, 석사], Certificate [[1, 2], [2], []], Family [["부", "모"], ["배우자"], []], Rank [상병, 병장, 이병], Position [통신, 행정, 보급], Class [101, 102, 103], PE [201, 202, 203], Affiliation [301, 302, 303] );
                        # 신체검사기록 ( ID [201, 202, 203], Date ["2024-01-01", "2024-02-01", "2024-03-01"], Location [서울, 대구, 대전], Height [175.5, 180.0, 165.0], Weight [70.0, 75.0, 60.0], Eye_L [1.0, 0.8, 0.9], Eye_R [1.2, 1.0, 0.8], BloodType [A, B, O], Grade [1, 2, 3] );
                        # 병과 ( Class [101, 102, 103], ReqCert [["통신자격증"], ["행정자격증"], []], ReqGrade [1, 2, 3] );
                        # 자격증 ( ID [1, 2, 3], Name ["통신자격증", "행정자격증", "보급자격증"] );
                        # 군수 ( ID [401, 402, 403], TYPE_A ["장비", "물자", "식량"], TYPE_B ["통신", "행정", "보급"], TYPE_C ["안테나", "문서", "식량포대"], Storage [301, 302, 303] );
                        # 부대 ( ID [301, 302, 303], Longitude [127.1, 128.2, 126.9], Latitude [37.5, 35.1, 37.4], Commander [A001, A002, A003], SubordinateUnit [["302", "303"], ["301"], []], Organization ["본부", "사단", "여단"], Num [1000, 800, 500] );
                        #
                        ### Foreign key information of SQLite tables, used for table joins:
                        #
                        # 인적정보 ( Class ) REFERENCES 병과 ( Class );
                        # 인적정보 ( PE ) REFERENCES 신체검사기록 ( ID );
                        # 인적정보 ( Affiliation ) REFERENCES 부대 ( ID );
                        # 군수 ( Storage ) REFERENCES 부대 ( ID );

                        ### Let’s first understand the problem, extract relevant variables and their corresponding numerals, and make and device a complete plan. Then, let’s carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and commonsense), solve the problem step by step, and show the answer.

                        ### Response Guidelines
                        # 1. If the provided context is sufficient, please generate a valid query without any explanations for the question. The query should start with a comment containing the question being asked.
                        # 2. If the provided context is insufficient, please explain why it can't be generated.
                        # 3. Please use the most relevant tabel(s).
                        # 4. Please format the query before responding.
                        # 5. Please always respond with the following format.

                        ### Response Format
                        {{
                                "query": "A generated SQL query when context is sufficient.",
                                "explanation": "An explanation of failing to generate the query."
                        }}

                        ### Question : "In a military database, there are 2,000 soldiers. 25% of them are assigned to units where the organization is '본부'. Of the remaining soldiers, 50% are affiliated with units whose commanders have blood type 'A' and have passed a physical examination with a Grade of 1 or 2. Among the rest, 30% are affiliated with units storing supplies where TYPE_A = ‘장비’ and TYPE_B = ‘통신’. The remaining soldiers are affiliated with units where the commander's position is '통신' and the unit's total number of soldiers (Num) is greater than 1,000. What percentage of the total soldiers are affiliated with units where TYPE_A = ‘장비’ and TYPE_B = ‘통신’, and what percentage are affiliated with units where the commander's position is '통신' and the total number of soldiers exceeds 1,000?"
                        """
        }
    ]
)

print(response.choices[0].message.content)

