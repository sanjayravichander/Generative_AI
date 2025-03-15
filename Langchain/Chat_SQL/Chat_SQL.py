import sqlite3

# Connect to SQLite database (creates if it doesn't exist)
connection = sqlite3.connect('STUDENTS.db')
cursor = connection.cursor()

# Create table if it doesn't exist
table_info = """
CREATE TABLE IF NOT EXISTS STUDENT (
    NAME VARCHAR(25),
    CLASS VARCHAR(25),
    SECTION VARCHAR(25),
    MARKS INT
);
"""
cursor.execute(table_info)

# Insert records using executemany for efficiency
cursor.executemany(
    '''INSERT INTO STUDENT VALUES (?, ?, ?, ?)''',
    [
        ('Krish', 'Data Science', 'A', 90),
        ('Sunita', 'Data Engineer', 'B', 100),
        ('Vandana', 'Data Analyst', 'A', 86),
        ('Dorothy', 'Data Science', 'B', 90),
        ('Sanjay', 'Data Science', 'A', 65),
        ('Sanjith', 'Data Engineer', 'A', 78),
        ('Sujith', 'Data Engineer', 'B', 80),
        ('Kamal', 'Data Analyst', 'A', 90),
        ('Karthick', 'Data Analyst', 'B', 100),
        ('Narean', 'Data Science', 'A', 86),
    ]
)

# Commit changes
connection.commit()

# Display all records from the correct table name
print("The inserted records are: ")
data = cursor.execute('''SELECT * FROM STUDENT''')
for row in data:
    print(row)

# Close connection
connection.close()
