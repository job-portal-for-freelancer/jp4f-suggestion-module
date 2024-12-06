import pyodbc 

cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=34.87.95.20,1433;Database=JP4F;UID=sa;PWD=@dmin123')

cursor = cnxn.cursor()
cursor.execute("SELECT p.Id, p.JobTittle, p.JobDesciption, STRING_AGG(s.SkillName, ', ') AS Skills FROM Project p LEFT JOIN ProjectSkill ps ON ps.ProjectsId = p.Id LEFT JOIN Skill s ON ps.SkillsId = s.Id GROUP BY p.Id, p.JobTittle, p.JobDesciption;")


leng = 0
for row in cursor:
    print('row = %r' % (row,))
    print("======Start=====")
    print(f"{row[1]} {row[2]} {row[3]}")
    processed_row = tuple(str(field).replace('-', '.').replace('\n', ' ') if isinstance(field, str) else field for field in row)
    print("-------")
    print(processed_row)
    print(f"{processed_row[1]} {processed_row[2]} {processed_row[3]}")
    print("=======End=====")

    leng += 1
    if leng == 1:
        break

