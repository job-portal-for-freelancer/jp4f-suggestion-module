import pyodbc 

cnxn = pyodbc.connect(
    "Driver={ODBC Driver 17 for SQL Server};"
    "Server=34.87.95.20,1433;"
    "Database=JP4F;"
    "uid=sa;"
    "pwd=@dmin123;"
)


cursor = cnxn.cursor()
cursor.execute('SELECT * FROM Project')

for row in cursor:
    print('row = %r' % (row,))