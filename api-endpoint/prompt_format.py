prompt_structure = '''As a 20-year experiences HR with a lots of experience of reading JDs and write JDs.
Answer the Questions. Use the information provided in the query to answer the question.
You can use the information to answer the question.

Query:
{prompt}

Questions:
{system}

The output format expected as the follow instructions:
{instruction}
'''

# Query - sẽ là input prompt lấy từ user
# System - sẽ là yêu cầu của mày
# Instruction - sẽ là hướng dẫn cho cái format của output
system_prompt = '''
Base on the query, write me a job description, must have Jop Title, Job description ,requirement, and salary.
'''

instruction = '''
return as a human-language, and language base on the query, return me with the HTML tag

'''



# Tham khảo cái system
system_prompt_cv = '''
Let's think step by step. The term Curriculum Vitae or Resume is annotated by CV
CV details might be out of order or incomplete. Some word might be broken when converting document.

Read and find the keywords of the provided CV, keywords must be simple, clearly, easy to understand, and specific, NOT a broad categories.
The keywords should be found inside the skills, experiements, projects, or where that appear the skill of candidates CV. Ignore overview, candidate introduction. Ignore company, organization name as a keyword of CV.

Else, give a mark for each found keywords of the CV, must in the range from 1 to 10, Mark should be an integer number.
There are some rules for giving mark:
- If the keyword is mentioned in a school project, hobbies project or just mentioned, the mark must be around 1-3. This range (1 to 3) is proportional to the appearance of the keyword.
- If the keyword is mentioned in a comany project, freelance project, or associated with such word `middle`, etc, the mark must be around 4-6. This range (4 to 6) is proportional to the appearance of the keyword.
- If the keyword is mentioned in a large comany project, more freelance projects, or associated with such word `senior`, `expert`, etc, the mark must be around 7-9. This range (7-9) is proportional to the appearance of the keyword.
- If the keyword is mentioned and satisfy the above condition, and associated with such degree like professor, PhD, Doctor, etc, the mark must be 10.

Put these found keywords of CV into the corresponding criteria mention in below instruction, It's ok if there is a criteria not have any keyword. It's ok if found keyword not match any criteria.
The keyword must be strongly related to the criteria. Not mis-understanding between school degree and candidate skills. Not mis-understanding between company name, degree and candidate skills.

Must respone promptly, accurately, and professionally. No yapping.
'''

# Tham khảo cái Instruction
instruce = 'Extract keywords and the responding score of each criteria name. This field can be empty. This field is a dictionary contains a string as key, a number as value, not any other types.'






